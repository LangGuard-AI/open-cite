"""
AWS Bedrock discovery plugin for OpenCITE.

This plugin discovers AI usage in AWS Bedrock:
- Available foundation models
- Custom/fine-tuned models
- Provisioned throughput
- Model invocations (via CloudTrail/CloudWatch)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading

from open_cite.core import BaseDiscoveryPlugin
from open_cite.plugins.aws.base import AWSClientMixin

logger = logging.getLogger(__name__)


class AWSBedrockPlugin(AWSClientMixin, BaseDiscoveryPlugin):
    """
    AWS Bedrock discovery plugin.

    Discovers:
    - Available foundation models (Claude, Llama, Titan, etc.)
    - Custom/fine-tuned models
    - Provisioned throughput configurations
    - Model invocations via CloudTrail events
    - Usage patterns via CloudWatch Logs (if enabled)

    Supports both inventory discovery (what exists) and usage discovery
    (what's actually being used) through CloudTrail and CloudWatch.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        profile: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        role_arn: Optional[str] = None,
        cloudwatch_log_group: Optional[str] = None,
    ):
        """
        Initialize the AWS Bedrock plugin.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS profile name
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            session_token: AWS session token (for temporary credentials)
            role_arn: IAM role ARN to assume
            cloudwatch_log_group: CloudWatch log group for Bedrock invocation logs
        """
        AWSClientMixin.__init__(
            self,
            region=region,
            profile=profile,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            role_arn=role_arn,
        )

        self.cloudwatch_log_group = cloudwatch_log_group

        # Cache for discovered data
        self._models_cache: Dict[str, Dict[str, Any]] = {}
        self._usage_cache: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "aws_bedrock"

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to AWS Bedrock.

        Returns:
            Dict with connection status and details
        """
        try:
            client = self._get_client("bedrock")
            # Simple API call to verify credentials
            response = client.list_foundation_models(byOutputModality="TEXT")
            model_count = len(response.get("modelSummaries", []))

            account_id = self.get_account_id()

            return {
                "success": True,
                "region": self.region,
                "account_id": account_id,
                "foundation_models_available": model_count,
                "message": "Successfully connected to AWS Bedrock",
            }
        except Exception as e:
            return {
                "success": False,
                "region": self.region,
                "error": str(e),
                "message": "Failed to connect to AWS Bedrock",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List AWS Bedrock assets.

        Supported asset types:
        - "model": Foundation models available in Bedrock
        - "custom_model": Custom/fine-tuned models
        - "provisioned_throughput": Provisioned model throughput
        - "invocation": Model invocations from CloudTrail

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "model":
                return self._list_foundation_models(**kwargs)
            elif asset_type == "custom_model":
                return self._list_custom_models(**kwargs)
            elif asset_type == "provisioned_throughput":
                return self._list_provisioned_throughput(**kwargs)
            elif asset_type == "invocation":
                return self._list_invocations_from_cloudtrail(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: model, custom_model, provisioned_throughput, invocation"
                )

    def _list_foundation_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List available Bedrock foundation models.

        Returns:
            List of foundation models with metadata
        """
        try:
            client = self._get_client("bedrock")
            response = client.list_foundation_models()

            models = []
            for model in response.get("modelSummaries", []):
                model_info = {
                    "id": model["modelId"],
                    "name": model.get("modelName", model["modelId"]),
                    "provider": model.get("providerName", "unknown"),
                    "type": "foundation_model",
                    "discovery_source": "aws_bedrock_api",
                    "modalities": {
                        "input": model.get("inputModalities", []),
                        "output": model.get("outputModalities", []),
                    },
                    "streaming_supported": model.get("responseStreamingSupported", False),
                    "customization_supported": model.get("customizationsSupported", []),
                    "inference_types": model.get("inferenceTypesSupported", []),
                    "region": self.region,
                    "metadata": {
                        "model_arn": model.get("modelArn"),
                        "model_lifecycle": model.get("modelLifecycle", {}),
                    },
                }

                # Cache the model
                self._models_cache[model["modelId"]] = model_info
                models.append(model_info)

            logger.info(f"Discovered {len(models)} Bedrock foundation models")
            return models

        except Exception as e:
            logger.error(f"Failed to list foundation models: {e}")
            return []

    def _list_custom_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List custom/fine-tuned models.

        Returns:
            List of custom models
        """
        try:
            client = self._get_client("bedrock")
            response = client.list_custom_models()

            models = []
            for model in response.get("modelSummaries", []):
                model_info = {
                    "id": model.get("modelArn", model.get("modelName")),
                    "name": model.get("modelName"),
                    "base_model": model.get("baseModelArn"),
                    "base_model_name": model.get("baseModelName"),
                    "type": "custom_model",
                    "discovery_source": "aws_bedrock_api",
                    "customization_type": model.get("customizationType"),
                    "creation_time": model.get("creationTime").isoformat()
                        if model.get("creationTime") else None,
                    "region": self.region,
                }

                models.append(model_info)

            logger.info(f"Discovered {len(models)} custom models")
            return models

        except Exception as e:
            logger.error(f"Failed to list custom models: {e}")
            return []

    def _list_provisioned_throughput(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List provisioned model throughput configurations.

        Returns:
            List of provisioned throughput configs
        """
        try:
            client = self._get_client("bedrock")
            response = client.list_provisioned_model_throughputs()

            throughputs = []
            for pt in response.get("provisionedModelSummaries", []):
                throughput_info = {
                    "id": pt.get("provisionedModelArn"),
                    "name": pt.get("provisionedModelName"),
                    "model_arn": pt.get("modelArn"),
                    "foundation_model_arn": pt.get("foundationModelArn"),
                    "type": "provisioned_throughput",
                    "discovery_source": "aws_bedrock_api",
                    "status": pt.get("status"),
                    "model_units": pt.get("modelUnits"),
                    "desired_model_units": pt.get("desiredModelUnits"),
                    "creation_time": pt.get("creationTime").isoformat()
                        if pt.get("creationTime") else None,
                    "last_modified_time": pt.get("lastModifiedTime").isoformat()
                        if pt.get("lastModifiedTime") else None,
                    "commitment_duration": pt.get("commitmentDuration"),
                    "region": self.region,
                }

                throughputs.append(throughput_info)

            logger.info(f"Discovered {len(throughputs)} provisioned throughputs")
            return throughputs

        except Exception as e:
            logger.error(f"Failed to list provisioned throughputs: {e}")
            return []

    def _list_invocations_from_cloudtrail(
        self,
        days: int = 7,
        max_results: int = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Discover model invocations from CloudTrail.

        This shows which Bedrock models are actually being USED,
        not just what's available.

        Args:
            days: Number of days to look back (default: 7)
            max_results: Maximum number of events to return

        Returns:
            List of invocation events
        """
        try:
            client = self._get_client("cloudtrail")

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            invocations = []
            events_processed = 0

            # Look for InvokeModel and InvokeModelWithResponseStream events
            for event_name in ["InvokeModel", "InvokeModelWithResponseStream"]:
                try:
                    paginator = client.get_paginator("lookup_events")

                    for page in paginator.paginate(
                        LookupAttributes=[
                            {"AttributeKey": "EventName", "AttributeValue": event_name}
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        PaginationConfig={"MaxItems": max_results}
                    ):
                        for event in page.get("Events", []):
                            events_processed += 1

                            try:
                                event_data = json.loads(event.get("CloudTrailEvent", "{}"))
                            except json.JSONDecodeError:
                                continue

                            request_params = event_data.get("requestParameters", {})
                            user_identity = event_data.get("userIdentity", {})

                            invocation = {
                                "id": event.get("EventId"),
                                "event_name": event_name,
                                "model_id": request_params.get("modelId"),
                                "timestamp": event.get("EventTime").isoformat()
                                    if event.get("EventTime") else None,
                                "type": "bedrock_invocation",
                                "discovery_source": "cloudtrail",
                                "user_arn": user_identity.get("arn"),
                                "user_type": user_identity.get("type"),
                                "principal_id": user_identity.get("principalId"),
                                "source_ip": event_data.get("sourceIPAddress"),
                                "user_agent": event_data.get("userAgent"),
                                "region": self.region,
                                "metadata": {
                                    "event_source": event_data.get("eventSource"),
                                    "aws_region": event_data.get("awsRegion"),
                                    "error_code": event_data.get("errorCode"),
                                    "error_message": event_data.get("errorMessage"),
                                },
                            }

                            invocations.append(invocation)

                            if len(invocations) >= max_results:
                                break

                except Exception as e:
                    logger.warning(f"Error querying CloudTrail for {event_name}: {e}")
                    continue

            logger.info(
                f"Discovered {len(invocations)} Bedrock invocations "
                f"(processed {events_processed} CloudTrail events)"
            )
            return invocations

        except Exception as e:
            logger.error(f"Failed to list invocations from CloudTrail: {e}")
            return []

    def _list_invocations_from_cloudwatch_logs(
        self,
        hours: int = 24,
        max_results: int = 1000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Discover model invocations from CloudWatch Logs.

        Requires Bedrock model invocation logging to be enabled.

        Args:
            hours: Number of hours to look back
            max_results: Maximum number of log events

        Returns:
            List of invocation log events
        """
        if not self.cloudwatch_log_group:
            logger.warning("CloudWatch log group not configured for Bedrock logs")
            return []

        try:
            client = self._get_client("logs")

            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = end_time - (hours * 60 * 60 * 1000)

            invocations = []

            paginator = client.get_paginator("filter_log_events")

            for page in paginator.paginate(
                logGroupName=self.cloudwatch_log_group,
                startTime=start_time,
                endTime=end_time,
                limit=max_results,
            ):
                for event in page.get("events", []):
                    try:
                        log_data = json.loads(event.get("message", "{}"))
                    except json.JSONDecodeError:
                        continue

                    # Parse Bedrock invocation log format
                    if log_data.get("schemaType") == "ModelInvocationLog":
                        invocation = {
                            "id": event.get("eventId"),
                            "model_id": log_data.get("modelId"),
                            "timestamp": datetime.fromtimestamp(
                                event.get("timestamp", 0) / 1000
                            ).isoformat(),
                            "type": "bedrock_invocation",
                            "discovery_source": "cloudwatch_logs",
                            "operation": log_data.get("operation"),
                            "input_tokens": log_data.get("input", {}).get("inputTokenCount"),
                            "output_tokens": log_data.get("output", {}).get("outputTokenCount"),
                            "region": log_data.get("region", self.region),
                            "account_id": log_data.get("accountId"),
                            "request_id": log_data.get("requestId"),
                        }

                        invocations.append(invocation)

            logger.info(f"Discovered {len(invocations)} invocations from CloudWatch Logs")
            return invocations

        except Exception as e:
            logger.error(f"Failed to query CloudWatch Logs: {e}")
            return []

    def get_usage_by_model(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate usage statistics by model.

        Combines CloudTrail events to show which models are being used,
        by whom, and how frequently.

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping model_id to usage statistics
        """
        invocations = self._list_invocations_from_cloudtrail(days=days)

        usage = defaultdict(lambda: {
            "model_id": None,
            "invocation_count": 0,
            "unique_users": set(),
            "unique_ips": set(),
            "first_seen": None,
            "last_seen": None,
            "error_count": 0,
        })

        for inv in invocations:
            model_id = inv.get("model_id")
            if not model_id:
                continue

            usage[model_id]["model_id"] = model_id
            usage[model_id]["invocation_count"] += 1

            if inv.get("user_arn"):
                usage[model_id]["unique_users"].add(inv["user_arn"])

            if inv.get("source_ip"):
                usage[model_id]["unique_ips"].add(inv["source_ip"])

            if inv.get("metadata", {}).get("error_code"):
                usage[model_id]["error_count"] += 1

            # Update timestamps
            timestamp = inv.get("timestamp")
            if timestamp:
                if not usage[model_id]["first_seen"] or timestamp < usage[model_id]["first_seen"]:
                    usage[model_id]["first_seen"] = timestamp
                if not usage[model_id]["last_seen"] or timestamp > usage[model_id]["last_seen"]:
                    usage[model_id]["last_seen"] = timestamp

        # Convert sets to lists and add counts
        result = {}
        for model_id, stats in usage.items():
            result[model_id] = {
                **stats,
                "unique_users": list(stats["unique_users"]),
                "unique_user_count": len(stats["unique_users"]),
                "unique_ips": list(stats["unique_ips"]),
                "unique_ip_count": len(stats["unique_ips"]),
            }

        return result

    def get_usage_by_user(self, days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate usage statistics by user/principal.

        Shows which users/roles are using Bedrock and which models.

        Args:
            days: Number of days to analyze

        Returns:
            Dict mapping user ARN to usage statistics
        """
        invocations = self._list_invocations_from_cloudtrail(days=days)

        usage = defaultdict(lambda: {
            "user_arn": None,
            "user_type": None,
            "invocation_count": 0,
            "models_used": set(),
            "first_seen": None,
            "last_seen": None,
        })

        for inv in invocations:
            user_arn = inv.get("user_arn")
            if not user_arn:
                continue

            usage[user_arn]["user_arn"] = user_arn
            usage[user_arn]["user_type"] = inv.get("user_type")
            usage[user_arn]["invocation_count"] += 1

            if inv.get("model_id"):
                usage[user_arn]["models_used"].add(inv["model_id"])

            # Update timestamps
            timestamp = inv.get("timestamp")
            if timestamp:
                if not usage[user_arn]["first_seen"] or timestamp < usage[user_arn]["first_seen"]:
                    usage[user_arn]["first_seen"] = timestamp
                if not usage[user_arn]["last_seen"] or timestamp > usage[user_arn]["last_seen"]:
                    usage[user_arn]["last_seen"] = timestamp

        # Convert sets to lists
        result = {}
        for user_arn, stats in usage.items():
            result[user_arn] = {
                **stats,
                "models_used": list(stats["models_used"]),
                "model_count": len(stats["models_used"]),
            }

        return result

    def refresh_discovery(self):
        """Refresh all cached discovery data."""
        with self._lock:
            logger.info("Refreshing AWS Bedrock discovery...")
            self._models_cache.clear()
            self._usage_cache.clear()

            self._list_foundation_models()
            self._list_custom_models()
            self._list_provisioned_throughput()

            logger.info("AWS Bedrock discovery refresh complete")
