"""
Google Cloud discovery plugin for Open Cite.

This plugin discovers AI tools and models deployed in Google Cloud Platform,
including Vertex AI models, endpoints, and generative AI resources.
"""

import logging
import subprocess
import socket
import json
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from open_cite.core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)


class GoogleCloudPlugin(BaseDiscoveryPlugin):
    """
    Google Cloud discovery plugin.

    Discovers AI resources in Google Cloud Platform:
    - Vertex AI models (custom trained models)
    - Vertex AI endpoints (model serving endpoints)
    - Generative AI models (Gemini, PaLM, etc.)
    - Model deployments and versions
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        credentials: Optional[Any] = None,
    ):
        """
        Initialize the Google Cloud plugin.

        Args:
            project_id: GCP project ID (if None, uses default from environment)
            location: GCP region/location (default: us-central1)
            credentials: GCP credentials (if None, uses default from environment)
        """
        self.project_id = project_id
        self.location = location
        self.credentials = credentials

        # Lazy-load GCP clients
        self._aiplatform_client = None
        self._model_service_client = None
        self._endpoint_service_client = None
        self._prediction_service_client = None

        # Cache for discovered resources
        self._models_cache: Dict[str, Dict[str, Any]] = {}
        self._endpoints_cache: Dict[str, Dict[str, Any]] = {}
        self._deployments_cache: Dict[str, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        import threading
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Name of the plugin."""
        return "google_cloud"

    @property
    def supported_asset_types(self) -> Set[str]:
        """Asset types supported by this plugin."""
        return {"model", "endpoint", "deployment", "generative_model", "mcp_server"}

    def get_identification_attributes(self) -> List[str]:
        return ["gcp.project_id", "gcp.location"]

    def _get_aiplatform_client(self):
        """Get or create Vertex AI client (lazy initialization)."""
        if self._aiplatform_client is None:
            try:
                from google.cloud import aiplatform

                # Initialize Vertex AI SDK
                aiplatform.init(
                    project=self.project_id,
                    location=self.location,
                    credentials=self.credentials,
                )
                self._aiplatform_client = aiplatform
                logger.info(f"Initialized Vertex AI client for project {self.project_id}")
            except ImportError:
                logger.error(
                    "google-cloud-aiplatform not installed. "
                    "Install with: pip install google-cloud-aiplatform"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI client: {e}")
                raise

        return self._aiplatform_client

    def _get_model_service_client(self):
        """Get or create Model Service client."""
        if self._model_service_client is None:
            try:
                from google.cloud.aiplatform_v1 import ModelServiceClient
                from google.api_core import client_options

                # Use regional endpoint to match the location
                api_endpoint = f"{self.location}-aiplatform.googleapis.com"
                client_opts = client_options.ClientOptions(api_endpoint=api_endpoint)

                self._model_service_client = ModelServiceClient(
                    client_options=client_opts,
                    credentials=self.credentials
                )
                logger.info(f"Initialized Model Service client for {self.location}")
            except ImportError:
                logger.error("google-cloud-aiplatform not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Model Service client: {e}")
                raise

        return self._model_service_client

    def _get_endpoint_service_client(self):
        """Get or create Endpoint Service client."""
        if self._endpoint_service_client is None:
            try:
                from google.cloud.aiplatform_v1 import EndpointServiceClient
                from google.api_core import client_options

                # Use regional endpoint to match the location
                api_endpoint = f"{self.location}-aiplatform.googleapis.com"
                client_opts = client_options.ClientOptions(api_endpoint=api_endpoint)

                self._endpoint_service_client = EndpointServiceClient(
                    client_options=client_opts,
                    credentials=self.credentials
                )
                logger.info(f"Initialized Endpoint Service client for {self.location}")
            except ImportError:
                logger.error("google-cloud-aiplatform not installed")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Endpoint Service client: {e}")
                raise

        return self._endpoint_service_client

    def verify_connection(self) -> Dict[str, Any]:
        """
        Verify connection to Google Cloud.

        Returns:
            Dict with connection status
        """
        try:
            # Try to initialize the client
            aiplatform = self._get_aiplatform_client()

            # Try to list endpoints (this will fail if credentials are bad)
            # Note: We use endpoints instead of models because the model list API
            # requires location='global' which may not match self.location
            try:
                endpoint_service = self._get_endpoint_service_client()
                parent = f"projects/{self.project_id}/locations/{self.location}"
                # Just check if we can make the request (don't iterate)
                request = {"parent": parent, "page_size": 1}
                endpoint_service.list_endpoints(request=request)

                return {
                    "success": True,
                    "project_id": self.project_id,
                    "location": self.location,
                    "message": "Successfully connected to Google Cloud Vertex AI",
                }
            except Exception as e:
                return {
                    "success": False,
                    "project_id": self.project_id,
                    "location": self.location,
                    "error": str(e),
                    "message": "Failed to access Vertex AI resources",
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to initialize Google Cloud client",
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List Google Cloud AI assets.

        Supported asset types:
        - "model": Vertex AI models
        - "endpoint": Vertex AI endpoints
        - "deployment": Model deployments
        - "generative_model": Generative AI models (Gemini, PaLM, etc.)
        - "mcp_server": MCP servers running on Compute Engine instances

        Args:
            asset_type: Type of asset to list
            **kwargs: Additional filters

        Returns:
            List of assets
        """
        with self._lock:
            if asset_type == "model":
                return self._list_models(**kwargs)
            elif asset_type == "endpoint":
                return self._list_endpoints(**kwargs)
            elif asset_type == "deployment":
                return self._list_deployments(**kwargs)
            elif asset_type == "generative_model":
                return self._list_generative_models(**kwargs)
            elif asset_type == "mcp_server":
                return self._list_mcp_servers(**kwargs)
            else:
                raise ValueError(
                    f"Unsupported asset type: {asset_type}. "
                    f"Supported types: model, endpoint, deployment, generative_model, mcp_server"
                )

    def _list_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List Vertex AI models.

        Returns:
            List of models with metadata
        """
        try:
            aiplatform = self._get_aiplatform_client()

            # List all models in the project/location
            models = aiplatform.Model.list(
                location=self.location,
                project=self.project_id,
            )

            model_list = []
            for model in models:
                model_info = {
                    "id": model.resource_name,
                    "name": model.display_name,
                    "resource_name": model.resource_name,
                    "discovery_source": "google_cloud_api",
                    "type": "vertex_ai_model",
                    "created_time": model.create_time.isoformat() if hasattr(model, "create_time") and model.create_time else None,
                    "updated_time": model.update_time.isoformat() if hasattr(model, "update_time") and model.update_time else None,
                    "project_id": self.project_id,
                    "location": self.location,
                    "metadata": {},
                }

                # Add model version if available
                if hasattr(model, "version_id"):
                    model_info["version"] = model.version_id

                # Add labels if available
                if hasattr(model, "labels") and model.labels:
                    model_info["metadata"]["labels"] = dict(model.labels)

                # Add description if available
                if hasattr(model, "description") and model.description:
                    model_info["description"] = model.description

                # Add encryption spec if available
                if hasattr(model, "encryption_spec") and model.encryption_spec:
                    model_info["metadata"]["encryption"] = "enabled"

                # Cache the model
                self._models_cache[model.resource_name] = model_info

                model_list.append(model_info)

            logger.info(f"Discovered {len(model_list)} Vertex AI models")
            return model_list

        except Exception as e:
            logger.error(f"Failed to list Vertex AI models: {e}")
            return []

    def _list_endpoints(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List Vertex AI endpoints.

        Returns:
            List of endpoints with metadata
        """
        try:
            aiplatform = self._get_aiplatform_client()

            # List all endpoints in the project/location
            endpoints = aiplatform.Endpoint.list(
                location=self.location,
                project=self.project_id,
            )

            endpoint_list = []
            for endpoint in endpoints:
                endpoint_info = {
                    "id": endpoint.resource_name,
                    "name": endpoint.display_name,
                    "resource_name": endpoint.resource_name,
                    "discovery_source": "google_cloud_api",
                    "type": "vertex_ai_endpoint",
                    "created_time": endpoint.create_time.isoformat() if hasattr(endpoint, "create_time") and endpoint.create_time else None,
                    "updated_time": endpoint.update_time.isoformat() if hasattr(endpoint, "update_time") and endpoint.update_time else None,
                    "project_id": self.project_id,
                    "location": self.location,
                    "deployed_models": [],
                    "metadata": {},
                }

                # Add description if available
                if hasattr(endpoint, "description") and endpoint.description:
                    endpoint_info["description"] = endpoint.description

                # Add labels if available
                if hasattr(endpoint, "labels") and endpoint.labels:
                    endpoint_info["metadata"]["labels"] = dict(endpoint.labels)

                # Add network configuration if available
                if hasattr(endpoint, "network") and endpoint.network:
                    endpoint_info["metadata"]["network"] = endpoint.network

                # Add deployed models information
                if hasattr(endpoint, "deployed_models") and endpoint.deployed_models:
                    for deployed_model in endpoint.deployed_models:
                        deployment_info = {
                            "deployed_model_id": deployed_model.id,
                            "model": deployed_model.model if hasattr(deployed_model, "model") else None,
                            "display_name": deployed_model.display_name if hasattr(deployed_model, "display_name") else None,
                        }

                        # Add machine type if available
                        if hasattr(deployed_model, "dedicated_resources"):
                            resources = deployed_model.dedicated_resources
                            if hasattr(resources, "machine_spec"):
                                deployment_info["machine_type"] = resources.machine_spec.machine_type
                            if hasattr(resources, "min_replica_count"):
                                deployment_info["min_replicas"] = resources.min_replica_count
                            if hasattr(resources, "max_replica_count"):
                                deployment_info["max_replicas"] = resources.max_replica_count

                        endpoint_info["deployed_models"].append(deployment_info)

                # Cache the endpoint
                self._endpoints_cache[endpoint.resource_name] = endpoint_info

                endpoint_list.append(endpoint_info)

            logger.info(f"Discovered {len(endpoint_list)} Vertex AI endpoints")
            return endpoint_list

        except Exception as e:
            logger.error(f"Failed to list Vertex AI endpoints: {e}")
            return []

    def _list_deployments(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List model deployments (models deployed to endpoints).

        Returns:
            List of deployments
        """
        # First ensure we have endpoints
        if not self._endpoints_cache:
            self._list_endpoints()

        deployments = []
        for endpoint_name, endpoint_info in self._endpoints_cache.items():
            for deployed_model in endpoint_info.get("deployed_models", []):
                deployment = {
                    "id": f"{endpoint_name}/{deployed_model.get('deployed_model_id')}",
                    "endpoint_id": endpoint_name,
                    "endpoint_name": endpoint_info.get("name"),
                    "model_id": deployed_model.get("model"),
                    "deployed_model_id": deployed_model.get("deployed_model_id"),
                    "deployed_model_name": deployed_model.get("display_name"),
                    "discovery_source": "google_cloud_api",
                    "type": "vertex_ai_deployment",
                    "project_id": self.project_id,
                    "location": self.location,
                }

                # Add machine configuration
                if "machine_type" in deployed_model:
                    deployment["machine_type"] = deployed_model["machine_type"]
                if "min_replicas" in deployed_model:
                    deployment["min_replicas"] = deployed_model["min_replicas"]
                if "max_replicas" in deployed_model:
                    deployment["max_replicas"] = deployed_model["max_replicas"]

                deployments.append(deployment)

        logger.info(f"Discovered {len(deployments)} model deployments")
        return deployments

    def _list_generative_models(self, **kwargs) -> List[Dict[str, Any]]:
        """
        List available generative AI models.

        Returns:
            List of generative models (Gemini, PaLM, etc.)
        """
        # Predefined list of Google's generative AI models
        # These are the models available in Model Garden
        generative_models = [
            {
                "id": "gemini-pro",
                "name": "Gemini Pro",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "gemini",
                "capabilities": ["text_generation", "chat", "code_generation"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "gemini-pro-vision",
                "name": "Gemini Pro Vision",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "gemini",
                "capabilities": ["multimodal", "vision", "text_generation"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "gemini-1.5-pro",
                "name": "Gemini 1.5 Pro",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "gemini",
                "capabilities": ["text_generation", "chat", "long_context", "multimodal"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "gemini-1.5-flash",
                "name": "Gemini 1.5 Flash",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "gemini",
                "capabilities": ["text_generation", "chat", "fast_inference"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "text-bison",
                "name": "PaLM 2 Text Bison",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "palm",
                "capabilities": ["text_generation"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "chat-bison",
                "name": "PaLM 2 Chat Bison",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "palm",
                "capabilities": ["chat", "conversation"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "codechat-bison",
                "name": "PaLM 2 Code Chat Bison",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "palm",
                "capabilities": ["code_generation", "chat"],
                "project_id": self.project_id,
                "location": self.location,
            },
            {
                "id": "embedding-gecko",
                "name": "Text Embedding Gecko",
                "discovery_source": "google_cloud_api",
                "type": "generative_ai_model",
                "provider": "google",
                "model_family": "gecko",
                "capabilities": ["embeddings", "semantic_search"],
                "project_id": self.project_id,
                "location": self.location,
            },
        ]

        logger.info(f"Listed {len(generative_models)} generative AI models")
        return generative_models

    def _list_mcp_servers(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Discover MCP servers running on Compute Engine instances.

        Uses instance labels to identify MCP servers. Looks for labels:
        - mcp-server: Server name/type (e.g., "filesystem", "postgres")
        - mcp-transport: Transport type (e.g., "stdio", "http", "sse")
        - mcp-port: Port number (for HTTP/SSE servers)
        - mcp-command: Command to start the server (optional)

        Args:
            **kwargs: Additional filters (e.g., zone to scan)

        Returns:
            List of MCP servers discovered on Compute Engine instances
        """
        try:
            from google.cloud import compute_v1

            instances_client = compute_v1.InstancesClient(credentials=self.credentials)
            zones_client = compute_v1.ZonesClient(credentials=self.credentials)

            # Get all zones in the project
            zones_to_scan = kwargs.get("zones")
            if not zones_to_scan:
                # List all zones
                zones_request = compute_v1.ListZonesRequest(project=self.project_id)
                all_zones = zones_client.list(request=zones_request)
                zones_to_scan = [zone.name for zone in all_zones]

            mcp_servers = []
            instances_scanned = 0

            # Scan each zone for instances
            for zone in zones_to_scan:
                try:
                    request = compute_v1.ListInstancesRequest(
                        project=self.project_id,
                        zone=zone,
                    )

                    instances = instances_client.list(request=request)

                    for instance in instances:
                        instances_scanned += 1

                        # Check if instance has MCP-related labels
                        if not instance.labels:
                            continue

                        # Look for mcp-server label
                        if "mcp-server" not in instance.labels:
                            continue

                        server_name = instance.labels["mcp-server"]

                        # Build MCP server info
                        mcp_server = {
                            "id": f"{instance.name}-{server_name}",
                            "name": server_name,
                            "discovery_source": "gcp_compute_labels",
                            "instance_name": instance.name,
                            "instance_id": str(instance.id),
                            "zone": zone,
                            "project_id": self.project_id,
                            "status": instance.status,
                        }

                        # Get transport type from label (default: stdio)
                        transport = instance.labels.get("mcp-transport", "stdio")
                        mcp_server["transport"] = transport

                        # For HTTP/SSE servers, get port and build endpoint
                        if transport in ["http", "sse"]:
                            port = instance.labels.get("mcp-port", "3000")

                            # Get external IP if available
                            external_ip = None
                            if instance.network_interfaces:
                                for interface in instance.network_interfaces:
                                    if interface.access_configs:
                                        for access_config in interface.access_configs:
                                            if access_config.nat_i_p:
                                                external_ip = access_config.nat_i_p
                                                break

                            # Get internal IP
                            internal_ip = None
                            if instance.network_interfaces:
                                internal_ip = instance.network_interfaces[0].network_i_p

                            if external_ip:
                                mcp_server["endpoint"] = f"http://{external_ip}:{port}"
                                mcp_server["external_ip"] = external_ip
                            elif internal_ip:
                                mcp_server["endpoint"] = f"http://{internal_ip}:{port}"

                            if internal_ip:
                                mcp_server["internal_ip"] = internal_ip

                            mcp_server["port"] = port

                        # For stdio servers, get command if available
                        elif transport == "stdio":
                            if "mcp-command" in instance.labels:
                                mcp_server["command"] = instance.labels["mcp-command"]

                        # Add all MCP-related labels as metadata
                        mcp_labels = {
                            k: v
                            for k, v in instance.labels.items()
                            if k.startswith("mcp-")
                        }
                        mcp_server["labels"] = mcp_labels

                        # Add other useful instance metadata
                        mcp_server["metadata"] = {
                            "machine_type": instance.machine_type,
                            "created_time": instance.creation_timestamp,
                        }

                        # Add any other non-MCP labels
                        other_labels = {
                            k: v
                            for k, v in instance.labels.items()
                            if not k.startswith("mcp-")
                        }
                        if other_labels:
                            mcp_server["metadata"]["instance_labels"] = other_labels

                        mcp_servers.append(mcp_server)

                except Exception as e:
                    logger.warning(f"Failed to scan zone {zone}: {e}")
                    continue

            logger.info(
                f"Discovered {len(mcp_servers)} MCP servers "
                f"(scanned {instances_scanned} instances across {len(zones_to_scan)} zones)"
            )
            return mcp_servers

        except ImportError:
            logger.error(
                "google-cloud-compute not installed. "
                "Install with: pip install google-cloud-compute"
            )
            return []
        except Exception as e:
            logger.error(f"Failed to discover MCP servers on Compute Engine: {e}")
            return []

    def _scan_instance_ports(
        self,
        instance_name: str,
        zone: str,
        ports: List[int] = None,
        timeout: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Scan ports on a GCP instance to detect open services.

        Uses gcloud to get instance IP, then scans specified ports.

        Args:
            instance_name: Name of the Compute Engine instance
            zone: Zone where the instance is located
            ports: List of ports to scan (default: common MCP ports)
            timeout: Socket timeout in seconds (default: 0.5)

        Returns:
            Dict with instance info and open ports
        """
        if ports is None:
            # Default MCP-related ports
            ports = [3000, 3001, 3002, 8000, 8080, 8888, 5000, 9000]

        result = {
            "instance_name": instance_name,
            "zone": zone,
            "open_ports": [],
            "scanned_ports": ports,
            "scan_method": "socket",
        }

        try:
            # Get instance details using gcloud
            cmd = [
                "gcloud",
                "compute",
                "instances",
                "describe",
                instance_name,
                f"--zone={zone}",
                f"--project={self.project_id}",
                "--format=json",
            ]

            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )

            if process.returncode != 0:
                logger.warning(
                    f"Failed to describe instance {instance_name}: {process.stderr}"
                )
                result["error"] = process.stderr
                return result

            instance_data = json.loads(process.stdout)

            # Get external IP
            external_ip = None
            if "networkInterfaces" in instance_data:
                for interface in instance_data["networkInterfaces"]:
                    if "accessConfigs" in interface:
                        for access_config in interface["accessConfigs"]:
                            if "natIP" in access_config:
                                external_ip = access_config["natIP"]
                                break

            if not external_ip:
                logger.info(
                    f"Instance {instance_name} has no external IP, skipping port scan"
                )
                result["error"] = "No external IP"
                return result

            result["ip_address"] = external_ip
            result["status"] = instance_data.get("status", "UNKNOWN")

            # Scan ports
            open_ports = self._scan_ports(external_ip, ports, timeout)
            result["open_ports"] = open_ports

            logger.info(
                f"Scanned {instance_name}: {len(open_ports)} open ports found"
            )

        except subprocess.TimeoutExpired:
            logger.error(f"gcloud command timed out for {instance_name}")
            result["error"] = "gcloud timeout"
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse gcloud output for {instance_name}: {e}")
            result["error"] = "JSON parse error"
        except Exception as e:
            logger.error(f"Failed to scan ports on {instance_name}: {e}")
            result["error"] = str(e)

        return result

    def _scan_ports(
        self, ip_address: str, ports: List[int], timeout: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Scan a list of ports on an IP address.

        Args:
            ip_address: IP address to scan
            ports: List of port numbers
            timeout: Socket timeout in seconds

        Returns:
            List of open ports with details
        """
        open_ports = []

        def scan_port(port: int) -> Optional[Dict[str, Any]]:
            """Scan a single port."""
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip_address, port))
                sock.close()

                if result == 0:
                    # Port is open, try to identify service
                    service = self._identify_service(port)
                    return {
                        "port": port,
                        "state": "open",
                        "service": service,
                    }
            except Exception as e:
                logger.debug(f"Error scanning port {port} on {ip_address}: {e}")

            return None

        # Scan ports in parallel for speed
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(scan_port, port): port for port in ports}

            for future in as_completed(futures):
                port_result = future.result()
                if port_result:
                    open_ports.append(port_result)

        return sorted(open_ports, key=lambda x: x["port"])

    def _identify_service(self, port: int) -> str:
        """
        Identify common services by port number.

        Args:
            port: Port number

        Returns:
            Service name or 'unknown'
        """
        common_services = {
            22: "ssh",
            80: "http",
            443: "https",
            3000: "http-alt",
            3001: "http-alt",
            3002: "http-alt",
            5000: "http-alt",
            8000: "http-alt",
            8080: "http-proxy",
            8888: "http-alt",
            9000: "http-alt",
        }

        return common_services.get(port, "unknown")

    def discover_mcp_servers_by_port_scan(
        self,
        zones: Optional[List[str]] = None,
        ports: Optional[List[int]] = None,
        min_ports_open: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Discover potential MCP servers by scanning for open ports on instances.

        This method:
        1. Lists all Compute Engine instances
        2. Scans common MCP ports on each instance
        3. Returns instances with open ports that might be MCP servers

        Args:
            zones: List of zones to scan (None = all zones)
            ports: List of ports to scan (None = default MCP ports)
            min_ports_open: Minimum open ports to consider an instance (default: 1)

        Returns:
            List of instances with open ports
        """
        try:
            from google.cloud import compute_v1

            instances_client = compute_v1.InstancesClient(credentials=self.credentials)
            zones_client = compute_v1.ZonesClient(credentials=self.credentials)

            # Get zones to scan
            zones_to_scan = zones
            if not zones_to_scan:
                zones_request = compute_v1.ListZonesRequest(project=self.project_id)
                all_zones = zones_client.list(request=zones_request)
                zones_to_scan = [zone.name for zone in all_zones]

            discovered_servers = []

            # Scan each zone
            for zone in zones_to_scan:
                try:
                    request = compute_v1.ListInstancesRequest(
                        project=self.project_id,
                        zone=zone,
                    )

                    instances = instances_client.list(request=request)

                    # Scan each instance
                    for instance in instances:
                        # Only scan running instances
                        if instance.status != "RUNNING":
                            continue

                        logger.info(f"Scanning ports on {instance.name} in {zone}...")
                        scan_result = self._scan_instance_ports(
                            instance.name, zone, ports
                        )

                        # Check if enough ports are open
                        if len(scan_result.get("open_ports", [])) >= min_ports_open:
                            # Build server info
                            server_info = {
                                "id": f"{instance.name}-portscan",
                                "name": f"{instance.name}",
                                "discovery_source": "gcp_port_scan",
                                "instance_name": instance.name,
                                "instance_id": str(instance.id),
                                "zone": zone,
                                "project_id": self.project_id,
                                "status": instance.status,
                                "ip_address": scan_result.get("ip_address"),
                                "open_ports": scan_result.get("open_ports", []),
                                "scanned_ports": scan_result.get("scanned_ports", []),
                            }

                            # Try to infer MCP server type from open ports
                            open_port_numbers = [
                                p["port"] for p in scan_result.get("open_ports", [])
                            ]

                            # Infer transport and endpoint
                            if open_port_numbers:
                                # Prefer common HTTP ports
                                http_ports = [
                                    p for p in open_port_numbers if p in [3000, 8000, 8080]
                                ]
                                port = http_ports[0] if http_ports else open_port_numbers[0]

                                server_info["transport"] = "http"
                                server_info["port"] = port
                                server_info["endpoint"] = (
                                    f"http://{scan_result.get('ip_address')}:{port}"
                                )
                                server_info["inferred"] = True

                            # Add metadata
                            server_info["metadata"] = {
                                "machine_type": instance.machine_type,
                                "scan_time": datetime.utcnow().isoformat(),
                            }

                            # Check labels for MCP info
                            if instance.labels:
                                if "mcp-server" in instance.labels:
                                    server_info["name"] = instance.labels["mcp-server"]
                                    server_info["inferred"] = False

                                mcp_labels = {
                                    k: v
                                    for k, v in instance.labels.items()
                                    if k.startswith("mcp-")
                                }
                                if mcp_labels:
                                    server_info["labels"] = mcp_labels

                            discovered_servers.append(server_info)

                except Exception as e:
                    logger.warning(f"Failed to scan zone {zone}: {e}")
                    continue

            logger.info(
                f"Port scan discovered {len(discovered_servers)} potential MCP servers"
            )
            return discovered_servers

        except ImportError:
            logger.error("google-cloud-compute not installed")
            return []
        except Exception as e:
            logger.error(f"Failed to discover MCP servers by port scan: {e}")
            return []

    def refresh_discovery(self):
        """Refresh all cached discovery data."""
        with self._lock:
            logger.info("Refreshing Google Cloud discovery...")
            self._models_cache.clear()
            self._endpoints_cache.clear()
            self._deployments_cache.clear()

            # Re-discover all resources
            self._list_models()
            self._list_endpoints()
            self._list_deployments()

            logger.info("Google Cloud discovery refresh complete")
