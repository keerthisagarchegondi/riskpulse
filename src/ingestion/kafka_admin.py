"""Kafka admin client for topic management.

Creates and configures Kafka topics based on kafka_config.yaml.
Ensures topics exist with proper partitioning, replication,
and retention settings before producers/consumers start.
"""

from __future__ import annotations

from typing import Any

import structlog
import yaml
from confluent_kafka.admin import AdminClient, NewTopic, ConfigResource, ResourceType

from src.utils.config import get_settings

logger = structlog.get_logger(__name__)


class KafkaAdminError(Exception):
    """Raised when Kafka admin operations fail."""


class KafkaTopicManager:
    """Manages Kafka topic creation and configuration.

    Reads topic definitions from kafka_config.yaml and ensures
    they exist in the cluster with correct settings.
    """

    def __init__(self, bootstrap_servers: str | None = None) -> None:
        settings = get_settings()
        self._bootstrap_servers = bootstrap_servers or settings.get(
            "kafka.bootstrap_servers", "localhost:9092"
        )
        self._admin_client = AdminClient({
            "bootstrap.servers": self._bootstrap_servers,
            "client.id": "riskpulse-admin",
            "request.timeout.ms": 10000,
        })

    def create_topics(
        self,
        config_path: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, str]:
        """Create all topics defined in kafka_config.yaml.

        Args:
            config_path: Path to kafka config YAML. Uses default if None.
            dry_run: If True, only validate without creating.

        Returns:
            Dict mapping topic name -> status ('created', 'exists', 'error').
        """
        from pathlib import Path
        from src.utils.config import _get_project_root

        if config_path is None:
            config_path = str(_get_project_root() / "config" / "kafka_config.yaml")

        with open(config_path, "r") as f:
            kafka_config = yaml.safe_load(f)

        topics_config = kafka_config.get("topics", {})
        if not topics_config:
            logger.warning("no_topics_in_config", config_path=config_path)
            return {}

        # Get existing topics
        existing = self._get_existing_topics()

        results: dict[str, str] = {}
        new_topics: list[NewTopic] = []

        for topic_name, topic_settings in topics_config.items():
            if topic_name in existing:
                results[topic_name] = "exists"
                logger.debug("topic_already_exists", topic=topic_name)
                continue

            new_topic = NewTopic(
                topic=topic_name,
                num_partitions=topic_settings.get("partitions", 6),
                replication_factor=topic_settings.get("replication_factor", 1),
                config={
                    "retention.ms": str(topic_settings.get("retention_ms", 604800000)),
                    "cleanup.policy": topic_settings.get("cleanup_policy", "delete"),
                    "compression.type": "snappy",
                    "min.insync.replicas": "1",
                },
            )
            new_topics.append(new_topic)

        if dry_run:
            for topic in new_topics:
                results[topic.topic] = "would_create"
            logger.info("dry_run_complete", topics_to_create=len(new_topics))
            return results

        if not new_topics:
            logger.info("all_topics_exist")
            return results

        # Create topics
        futures = self._admin_client.create_topics(new_topics, operation_timeout=30)

        for topic_name, future in futures.items():
            try:
                future.result()
                results[topic_name] = "created"
                logger.info("topic_created", topic=topic_name)
            except Exception as e:
                results[topic_name] = f"error: {e}"
                logger.error("topic_creation_failed", topic=topic_name, error=str(e))

        return results

    def delete_topics(self, topic_names: list[str]) -> dict[str, str]:
        """Delete specified topics from the cluster.

        Args:
            topic_names: List of topic names to delete.

        Returns:
            Dict mapping topic name -> status.
        """
        results: dict[str, str] = {}
        futures = self._admin_client.delete_topics(topic_names, operation_timeout=30)

        for topic_name, future in futures.items():
            try:
                future.result()
                results[topic_name] = "deleted"
                logger.info("topic_deleted", topic=topic_name)
            except Exception as e:
                results[topic_name] = f"error: {e}"
                logger.error("topic_deletion_failed", topic=topic_name, error=str(e))

        return results

    def _get_existing_topics(self) -> set[str]:
        """Get list of existing topics from the cluster."""
        metadata = self._admin_client.list_topics(timeout=10)
        return set(metadata.topics.keys())

    def describe_topics(self, topic_names: list[str] | None = None) -> dict[str, Any]:
        """Get metadata for topics.

        Args:
            topic_names: Specific topics to describe. None for all.

        Returns:
            Dict with topic metadata.
        """
        metadata = self._admin_client.list_topics(timeout=10)
        result = {}

        topics = topic_names or list(metadata.topics.keys())
        for name in topics:
            if name in metadata.topics:
                topic_meta = metadata.topics[name]
                result[name] = {
                    "partitions": len(topic_meta.partitions),
                    "error": str(topic_meta.error) if topic_meta.error else None,
                }

        return result

    def ensure_topics_exist(self) -> None:
        """Ensure all configured topics exist. Create missing ones.

        This is intended to be called at application startup.
        """
        results = self.create_topics()
        created = [t for t, s in results.items() if s == "created"]
        existing = [t for t, s in results.items() if s == "exists"]
        errors = [t for t, s in results.items() if s.startswith("error")]

        logger.info(
            "topic_setup_complete",
            created=len(created),
            existing=len(existing),
            errors=len(errors),
        )

        if errors:
            error_details = {t: s for t, s in results.items() if s.startswith("error")}
            raise KafkaAdminError(
                f"Failed to create topics: {error_details}"
            )
