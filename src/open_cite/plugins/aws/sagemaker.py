"""
AWS SageMaker discovery plugin for OpenCITE.

This plugin discovers ML assets in AWS SageMaker:
- Endpoints (deployed models)
- Models
- Model registry packages
- Training jobs
- Endpoint invocation metrics
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


class AWSSageMakerPlugin(AWSClientMixin, BaseDiscoveryPlugin):
    """
    AWS SageMaker discovery plugin.

    Discovers:
    - Endpoints: Deployed model serving endpoints
    - Models: Model artifacts registered in SageMaker
    - Model Packages: Model registry entries
    - Training Jobs: ML training job history
    - Endpoint Usage: Invocation metrics from CloudWatch

    Supports both inventory discovery (what exists) and usage discovery
    (invocation counts, latency metrics) through CloudWatch.
    """

    plugin_type = "aws_sagemaker"

    def __init__(
        self,
        region: Optional[str] = None,
        profile: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        role_arn: Optional[str] = None,
        instance_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the AWS SageMaker plugin.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS profile name
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            session_token: AWS session token (for temporary credentials)
            role_arn: IAM role ARN to assume
            instance_id: Unique identifier for this plugin instance
            display_name: Human-readable name for this instance
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
        BaseDiscoveryPlugin.__init__(
            self,
            instance_id=instance_id,
            display_name=display_name,
        )

        # Cache for discovered data
        self._endpoints_cache: Dict[str, Dict[str, Any]] = {}
        self._models_cache: Dict[str, Dict[str, Any]] = {}

        self._lock = threading.Lock()

    @property
    def supported_asset_types(self):
        return {"endpoint", "model", "model_package", "training_job"}

    def get_identification_attributes(self) -> List[str]:
        return ["aws.sagemaker.endpoint_name", "aws.sagemaker.region", "aws.account_id"]

    def get_config(self) -> Dict[str, Any]:
        """Return plugin configuration (sensitive values masked)."""
        return {
            "region": self.region,
            "profile": self.profile,
            "access_key_id": "****" if self.access_key_id else None,
        }

    @classmethod
    def plugin_metadata(cls) -> Dict[str, Any]:
        return {
            "name": "AWS SageMaker",
            "description": "Discovers endpoints, models, model packages, and training jobs in AWS SageMaker",
            "required_fields": {
                "region": {"label": "AWS Region", "default": "us-east-1", "required": False},
                "profile": {"label": "AWS Profile", "default": "", "required": False},
                "access_key_id": {"label": "Access Key ID", "default": "", "required": False},
                "secret_access_key": {"label": "Secret Access Key", "default": "", "required": False, "type": "password"},
                "role_arn": {"label": "Role ARN (optional)", "default": "", "required": False},
            },
            "env_vars": ["AWS_REGION", "AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        }

    @classmethod
    def from_config(cls, config, instance_id=None, display_name=None, dependencies=None):
        return cls(
            region=config.get('region'),
            profile=config.get('profile'),
            access_key_id=config.get('access_key_id'),
            secret_access_key=config.get('secret_access_key'),
            session_token=config.get('session_token'),
            role_arn=config.get('role_arn'),
            instance_id=instance_id,
            display_name=display_name,
        )

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to AWS SageMaker.

        Returns:
            Dict with connection status
        """
        try:
            client = self._get_client("sagemaker")
            # Simple API call to verify credentials
            response = client.list_endpoints(MaxResults=1)

            account_id = self.get_account_id()

            return {
                "success": True,
                "region": self.region,
                "account_id": account_id,
                "message": "Successfully connected to AWS SageMaker",
            }
        except Exception as e:
            return {
                "success": False,
                "region": self.region,
                "error": str(e),
                "message": "Failed to connect to AWS SageMaker",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List AWS SageMaker assets.

        Supported asset types:
        - "endpoint": Model serving endpoints
        - "model": Registered models
        - "model_package": Model registry packages
        - "training_job": Training jobs

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "endpoint":
                return self._list_endpoints(**kwargs)
            elif asset_type == "model":
                return self._list_models(**kwargs)
            elif asset_type == "model_package":
                return self._list_model_packages(**kwargs)
            elif asset_type == "training_job":
                return self._list_training_jobs(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: endpoint, model, model_package, training_job"
                )

    def _list_endpoints(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List SageMaker endpoints.

        Returns:
            List of endpoints with metadata
        """
        try:
            client = self._get_client("sagemaker")

            endpoints = []
            paginator = client.get_paginator("list_endpoints")

            for page in paginator.paginate():
                for ep in page.get("Endpoints", []):
                    endpoint_name = ep["EndpointName"]

                    # Get detailed endpoint info
                    endpoint_info = {
                        "id": ep["EndpointArn"],
                        "name": endpoint_name,
                        "type": "sagemaker_endpoint",
                        "discovery_source": "aws_sagemaker_api",
                        "status": ep.get("EndpointStatus"),
                        "creation_time": ep.get("CreationTime").isoformat()
                            if ep.get("CreationTime") else None,
                        "last_modified": ep.get("LastModifiedTime").isoformat()
                            if ep.get("LastModifiedTime") else None,
                        "region": self.region,
                        "metadata": {},
                    }

                    # Try to get endpoint config details
                    try:
                        details = client.describe_endpoint(EndpointName=endpoint_name)
                        config_name = details.get("EndpointConfigName")

                        if config_name:
                            config = client.describe_endpoint_config(
                                EndpointConfigName=config_name
                            )

                            # Extract production variants info
                            variants = config.get("ProductionVariants", [])
                            if variants:
                                variant = variants[0]
                                endpoint_info["instance_type"] = variant.get("InstanceType")
                                endpoint_info["model_name"] = variant.get("ModelName")
                                endpoint_info["initial_instance_count"] = variant.get(
                                    "InitialInstanceCount"
                                )
                                endpoint_info["variant_name"] = variant.get("VariantName")

                            endpoint_info["metadata"]["endpoint_config_name"] = config_name
                            endpoint_info["metadata"]["production_variants_count"] = len(variants)

                    except Exception as e:
                        logger.debug(f"Could not get details for endpoint {endpoint_name}: {e}")

                    # Cache and add to list
                    self._endpoints_cache[endpoint_name] = endpoint_info
                    endpoints.append(endpoint_info)

            logger.info(f"Discovered {len(endpoints)} SageMaker endpoints")
            return endpoints

        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []

    def _list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List SageMaker models.

        Returns:
            List of models
        """
        try:
            client = self._get_client("sagemaker")

            models = []
            paginator = client.get_paginator("list_models")

            for page in paginator.paginate():
                for model in page.get("Models", []):
                    model_name = model["ModelName"]

                    model_info = {
                        "id": model["ModelArn"],
                        "name": model_name,
                        "type": "sagemaker_model",
                        "discovery_source": "aws_sagemaker_api",
                        "creation_time": model.get("CreationTime").isoformat()
                            if model.get("CreationTime") else None,
                        "region": self.region,
                        "metadata": {},
                    }

                    # Try to get model details
                    try:
                        details = client.describe_model(ModelName=model_name)

                        # Extract container info
                        primary_container = details.get("PrimaryContainer", {})
                        if primary_container:
                            model_info["container_image"] = primary_container.get("Image")
                            model_info["model_data_url"] = primary_container.get("ModelDataUrl")

                        # Extract execution role
                        model_info["execution_role"] = details.get("ExecutionRoleArn")

                        # Check for multi-model or inference component
                        model_info["metadata"]["enable_network_isolation"] = details.get(
                            "EnableNetworkIsolation", False
                        )

                    except Exception as e:
                        logger.debug(f"Could not get details for model {model_name}: {e}")

                    self._models_cache[model_name] = model_info
                    models.append(model_info)

            logger.info(f"Discovered {len(models)} SageMaker models")
            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def _list_model_packages(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List model registry packages.

        Returns:
            List of model packages
        """
        try:
            client = self._get_client("sagemaker")

            packages = []

            # First, list model package groups
            try:
                groups_paginator = client.get_paginator("list_model_package_groups")

                for groups_page in groups_paginator.paginate():
                    for group in groups_page.get("ModelPackageGroupSummaryList", []):
                        group_name = group["ModelPackageGroupName"]

                        # List packages in this group
                        try:
                            pkg_paginator = client.get_paginator("list_model_packages")

                            for pkg_page in pkg_paginator.paginate(
                                ModelPackageGroupName=group_name
                            ):
                                for pkg in pkg_page.get("ModelPackageSummaryList", []):
                                    package_info = {
                                        "id": pkg["ModelPackageArn"],
                                        "name": pkg.get("ModelPackageName",
                                            pkg["ModelPackageArn"].split("/")[-1]),
                                        "group": group_name,
                                        "group_description": group.get("ModelPackageGroupDescription"),
                                        "type": "model_package",
                                        "discovery_source": "aws_sagemaker_api",
                                        "status": pkg.get("ModelPackageStatus"),
                                        "approval_status": pkg.get("ModelApprovalStatus"),
                                        "creation_time": pkg.get("CreationTime").isoformat()
                                            if pkg.get("CreationTime") else None,
                                        "version": pkg.get("ModelPackageVersion"),
                                        "region": self.region,
                                    }

                                    packages.append(package_info)

                        except Exception as e:
                            logger.warning(f"Could not list packages in group {group_name}: {e}")

            except Exception as e:
                logger.warning(f"Could not list model package groups: {e}")

            # Also list unversioned model packages (not in groups)
            try:
                unversioned_paginator = client.get_paginator("list_model_packages")

                for page in unversioned_paginator.paginate(
                    ModelPackageType="Unversioned"
                ):
                    for pkg in page.get("ModelPackageSummaryList", []):
                        package_info = {
                            "id": pkg["ModelPackageArn"],
                            "name": pkg.get("ModelPackageName",
                                pkg["ModelPackageArn"].split("/")[-1]),
                            "group": None,
                            "type": "model_package",
                            "discovery_source": "aws_sagemaker_api",
                            "status": pkg.get("ModelPackageStatus"),
                            "approval_status": pkg.get("ModelApprovalStatus"),
                            "creation_time": pkg.get("CreationTime").isoformat()
                                if pkg.get("CreationTime") else None,
                            "region": self.region,
                        }

                        packages.append(package_info)

            except Exception as e:
                logger.debug(f"Could not list unversioned packages: {e}")

            logger.info(f"Discovered {len(packages)} model packages")
            return packages

        except Exception as e:
            logger.error(f"Failed to list model packages: {e}")
            return []

    def _list_training_jobs(self, days: int = 30, **kwargs) -> List[Dict[str, Any]]:
        """
        List recent training jobs.

        Args:
            days: Number of days to look back (default: 30)

        Returns:
            List of training jobs
        """
        try:
            client = self._get_client("sagemaker")

            jobs = []
            creation_time_after = datetime.utcnow() - timedelta(days=days)

            paginator = client.get_paginator("list_training_jobs")

            for page in paginator.paginate(CreationTimeAfter=creation_time_after):
                for job in page.get("TrainingJobSummaries", []):
                    job_name = job["TrainingJobName"]

                    job_info = {
                        "id": job["TrainingJobArn"],
                        "name": job_name,
                        "type": "training_job",
                        "discovery_source": "aws_sagemaker_api",
                        "status": job.get("TrainingJobStatus"),
                        "secondary_status": job.get("SecondaryStatus"),
                        "creation_time": job.get("CreationTime").isoformat()
                            if job.get("CreationTime") else None,
                        "end_time": job.get("TrainingEndTime").isoformat()
                            if job.get("TrainingEndTime") else None,
                        "last_modified": job.get("LastModifiedTime").isoformat()
                            if job.get("LastModifiedTime") else None,
                        "region": self.region,
                        "metadata": {},
                    }

                    # Try to get job details for more info
                    try:
                        details = client.describe_training_job(TrainingJobName=job_name)

                        # Resource config
                        resource_config = details.get("ResourceConfig", {})
                        job_info["instance_type"] = resource_config.get("InstanceType")
                        job_info["instance_count"] = resource_config.get("InstanceCount")
                        job_info["volume_size_gb"] = resource_config.get("VolumeSizeInGB")

                        # Algorithm/image
                        algorithm_spec = details.get("AlgorithmSpecification", {})
                        job_info["training_image"] = algorithm_spec.get("TrainingImage")
                        job_info["training_input_mode"] = algorithm_spec.get("TrainingInputMode")

                        # Billable seconds
                        job_info["billable_seconds"] = details.get("BillableTimeInSeconds")

                        # Output location
                        output_config = details.get("OutputDataConfig", {})
                        job_info["output_path"] = output_config.get("S3OutputPath")

                    except Exception as e:
                        logger.debug(f"Could not get details for job {job_name}: {e}")

                    jobs.append(job_info)

            logger.info(f"Discovered {len(jobs)} training jobs (last {days} days)")
            return jobs

        except Exception as e:
            logger.error(f"Failed to list training jobs: {e}")
            return []

    def get_endpoint_invocation_metrics(
        self,
        endpoint_name: str,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get invocation metrics for a specific endpoint.

        Args:
            endpoint_name: Name of the endpoint
            days: Number of days to look back

        Returns:
            Dict with invocation metrics
        """
        try:
            client = self._get_client("cloudwatch")

            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            metrics = {
                "endpoint_name": endpoint_name,
                "period_days": days,
                "invocations": None,
                "model_latency_avg_ms": None,
                "overhead_latency_avg_ms": None,
                "errors_4xx": None,
                "errors_5xx": None,
            }

            dimensions = [{"Name": "EndpointName", "Value": endpoint_name}]

            # Get invocation count
            try:
                response = client.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName="Invocations",
                    Dimensions=dimensions,
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=days * 86400,  # Aggregate over entire period
                    Statistics=["Sum"],
                )
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    metrics["invocations"] = int(sum(dp["Sum"] for dp in datapoints))
            except Exception as e:
                logger.debug(f"Could not get Invocations metric: {e}")

            # Get model latency
            try:
                response = client.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName="ModelLatency",
                    Dimensions=dimensions,
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=days * 86400,
                    Statistics=["Average"],
                )
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    # Convert from microseconds to milliseconds
                    avg_latency = sum(dp["Average"] for dp in datapoints) / len(datapoints)
                    metrics["model_latency_avg_ms"] = round(avg_latency / 1000, 2)
            except Exception as e:
                logger.debug(f"Could not get ModelLatency metric: {e}")

            # Get overhead latency
            try:
                response = client.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName="OverheadLatency",
                    Dimensions=dimensions,
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=days * 86400,
                    Statistics=["Average"],
                )
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    avg_latency = sum(dp["Average"] for dp in datapoints) / len(datapoints)
                    metrics["overhead_latency_avg_ms"] = round(avg_latency / 1000, 2)
            except Exception as e:
                logger.debug(f"Could not get OverheadLatency metric: {e}")

            # Get 4xx errors
            try:
                response = client.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName="Invocation4XXErrors",
                    Dimensions=dimensions,
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=days * 86400,
                    Statistics=["Sum"],
                )
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    metrics["errors_4xx"] = int(sum(dp["Sum"] for dp in datapoints))
            except Exception as e:
                logger.debug(f"Could not get 4XX errors metric: {e}")

            # Get 5xx errors
            try:
                response = client.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName="Invocation5XXErrors",
                    Dimensions=dimensions,
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=days * 86400,
                    Statistics=["Sum"],
                )
                datapoints = response.get("Datapoints", [])
                if datapoints:
                    metrics["errors_5xx"] = int(sum(dp["Sum"] for dp in datapoints))
            except Exception as e:
                logger.debug(f"Could not get 5XX errors metric: {e}")

            return metrics

        except Exception as e:
            logger.error(f"Failed to get metrics for endpoint {endpoint_name}: {e}")
            return {"endpoint_name": endpoint_name, "error": str(e)}

    def get_all_endpoint_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get invocation metrics for all endpoints.

        Args:
            days: Number of days to look back

        Returns:
            List of endpoint metrics
        """
        # Ensure endpoints are loaded
        if not self._endpoints_cache:
            self._list_endpoints()

        metrics = []
        for endpoint_name in self._endpoints_cache:
            endpoint_metrics = self.get_endpoint_invocation_metrics(endpoint_name, days)
            metrics.append(endpoint_metrics)

        # Sort by invocations (most active first)
        metrics.sort(key=lambda x: x.get("invocations") or 0, reverse=True)

        return metrics

    def get_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a summary of SageMaker usage.

        Args:
            days: Number of days to analyze

        Returns:
            Summary dict with counts and top endpoints
        """
        endpoints = self._list_endpoints()
        models = self._list_models()
        packages = self._list_model_packages()
        jobs = self._list_training_jobs(days=days)

        # Get metrics for all endpoints
        endpoint_metrics = self.get_all_endpoint_metrics(days=days)

        total_invocations = sum(
            m.get("invocations") or 0 for m in endpoint_metrics
        )

        active_endpoints = [
            m for m in endpoint_metrics
            if m.get("invocations") and m["invocations"] > 0
        ]

        return {
            "summary": {
                "total_endpoints": len(endpoints),
                "active_endpoints": len(active_endpoints),
                "total_models": len(models),
                "total_model_packages": len(packages),
                "training_jobs_last_n_days": len(jobs),
                "total_invocations_last_n_days": total_invocations,
            },
            "period_days": days,
            "top_endpoints_by_invocations": active_endpoints[:10],
            "region": self.region,
        }

    def refresh_discovery(self):
        """Refresh all cached discovery data."""
        with self._lock:
            logger.info("Refreshing AWS SageMaker discovery...")
            self._endpoints_cache.clear()
            self._models_cache.clear()

            self._list_endpoints()
            self._list_models()

            logger.info("AWS SageMaker discovery refresh complete")
