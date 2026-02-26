"""
Base AWS plugin functionality for Open-CITE.

Provides shared authentication and client management for AWS plugins.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AWSClientMixin:
    """
    Mixin providing shared AWS authentication and client management.

    Supports multiple authentication methods:
    1. Explicit credentials (access_key_id, secret_access_key)
    2. AWS profile name
    3. IAM role (when running on AWS infrastructure)
    """

    def __init__(
        self,
        region: Optional[str] = None,
        profile: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        role_arn: Optional[str] = None,
    ):
        """
        Initialize AWS client mixin.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS profile name from ~/.aws/credentials
            access_key_id: Explicit AWS access key ID
            secret_access_key: Explicit AWS secret access key
            session_token: Optional session token for temporary credentials
            role_arn: Optional IAM role ARN to assume
        """
        self.region = region or "us-east-1"
        self.profile = profile
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.role_arn = role_arn

        self._session = None
        self._clients = {}

    def _get_boto3_session(self):
        """
        Get or create a boto3 session with configured credentials.

        Returns:
            boto3.Session instance
        """
        if self._session is not None:
            return self._session

        try:
            import boto3
        except ImportError:
            logger.error(
                "boto3 not installed. Install with: pip install boto3"
            )
            raise

        # Build session kwargs
        session_kwargs = {"region_name": self.region}

        if self.profile:
            session_kwargs["profile_name"] = self.profile
            logger.debug(f"Using AWS profile: {self.profile}")
        elif self.access_key_id and self.secret_access_key:
            session_kwargs["aws_access_key_id"] = self.access_key_id
            session_kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                session_kwargs["aws_session_token"] = self.session_token
            logger.debug("Using explicit AWS credentials")
        else:
            logger.debug("Using default AWS credential chain")

        self._session = boto3.Session(**session_kwargs)

        # Assume role if specified
        if self.role_arn:
            self._session = self._assume_role(self._session)

        return self._session

    def _assume_role(self, session) -> Any:
        """
        Assume an IAM role and return a new session.

        Args:
            session: Existing boto3 session

        Returns:
            New boto3 session with assumed role credentials
        """
        import boto3

        sts = session.client("sts")

        response = sts.assume_role(
            RoleArn=self.role_arn,
            RoleSessionName="opencite-discovery"
        )

        credentials = response["Credentials"]

        return boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=self.region
        )

    def _get_client(self, service_name: str) -> Any:
        """
        Get or create a boto3 client for a service.

        Args:
            service_name: AWS service name (e.g., 'bedrock', 'sagemaker')

        Returns:
            boto3 client for the service
        """
        if service_name not in self._clients:
            session = self._get_boto3_session()
            self._clients[service_name] = session.client(service_name)
            logger.debug(f"Created {service_name} client for region {self.region}")

        return self._clients[service_name]

    def get_account_id(self) -> Optional[str]:
        """
        Get the AWS account ID.

        Returns:
            AWS account ID or None if unable to retrieve
        """
        try:
            sts = self._get_client("sts")
            identity = sts.get_caller_identity()
            return identity.get("Account")
        except Exception as e:
            logger.warning(f"Could not get AWS account ID: {e}")
            return None
