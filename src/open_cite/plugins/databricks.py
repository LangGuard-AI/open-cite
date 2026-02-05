import os
import logging
from typing import List, Dict, Any, Optional, Set
import mlflow
from databricks.sdk import WorkspaceClient
from ..core import BaseDiscoveryPlugin

logger = logging.getLogger(__name__)

class DatabricksPlugin(BaseDiscoveryPlugin):
    """
    Databricks discovery plugin.
    """

    def __init__(self, host: Optional[str] = None, token: Optional[str] = None, warehouse_id: Optional[str] = None, http_client: Any = None):
        self.host = host or os.getenv("DATABRICKS_HOST")
        self.token = token or os.getenv("DATABRICKS_TOKEN")
        self.warehouse_id = warehouse_id or os.getenv("DATABRICKS_WAREHOUSE_ID")

        if not self.host or not self.token:
            raise ValueError("Databricks host and token must be provided or set in environment variables.")

        # Configure MLflow
        os.environ["DATABRICKS_HOST"] = self.host
        os.environ["DATABRICKS_TOKEN"] = self.token
        mlflow.set_tracking_uri("databricks")

        # Configure Databricks SDK
        self.workspace_client = WorkspaceClient(host=self.host, token=self.token)
        
        # Inject custom session if provided
        if http_client:
            self.http_client = http_client
            # Injecting session into WorkspaceClient's API client
            # Note: This relies on internal SDK structure (api_client._session)
            # which is common in generated SDKs but might need adjustment if SDK version differs.
            if hasattr(self.workspace_client, "api_client") and hasattr(self.workspace_client.api_client, "_session"):
                 self.workspace_client.api_client._session = http_client._session
            elif hasattr(self.workspace_client, "_api_client") and hasattr(self.workspace_client._api_client, "_session"):
                 self.workspace_client._api_client._session = http_client._session
            else:
                logger.warning("Could not inject custom HTTP session into Databricks WorkspaceClient")
        
        self.mlflow_client = mlflow.tracking.MlflowClient()

    @property
    def name(self) -> str:
        return "databricks"

    @property
    def supported_asset_types(self) -> Set[str]:
        return {"catalog", "schema", "table", "volume", "model", "function"}

    def get_identification_attributes(self) -> List[str]:
        return ["databricks.workspace_id", "databricks.catalog", "databricks.schema"]

    def _get_warehouse_id(self) -> Optional[str]:
        """
        Get SQL warehouse ID, auto-discovering if not configured.

        Returns:
            Warehouse ID or None if none available
        """
        # Return configured warehouse if available
        if self.warehouse_id:
            return self.warehouse_id

        # Try to auto-discover a running warehouse
        try:
            warehouses = list(self.workspace_client.warehouses.list())

            # Prefer running warehouses
            for wh in warehouses:
                if wh.state and wh.state.value == "RUNNING":
                    logger.info(f"Auto-discovered running SQL warehouse: {wh.name} ({wh.id})")
                    self.warehouse_id = wh.id
                    return wh.id

            # Fall back to any available warehouse
            if warehouses:
                wh = warehouses[0]
                logger.info(f"Auto-discovered SQL warehouse: {wh.name} ({wh.id}) - may need to start it")
                self.warehouse_id = wh.id
                return wh.id

            logger.warning("No SQL warehouses found in workspace")
            return None

        except Exception as e:
            logger.warning(f"Could not auto-discover SQL warehouse: {e}")
            return None

    def verify_connection(self) -> Dict[str, Any]:
        try:
            user = self.workspace_client.current_user.me()
            return {
                "success": True,
                "user": user.user_name,
                "active": user.active
            }
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_assets(self, asset_type: str, **kwargs) -> List[Dict[str, Any]]:
        if asset_type == "catalog":
            return self._list_catalogs()
        elif asset_type == "schema":
            return self._list_schemas(kwargs.get("catalog_name"))
        elif asset_type == "table":
            return self._list_tables(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        elif asset_type == "volume":
            return self._list_volumes(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        elif asset_type == "model":
            return self._list_models(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        elif asset_type == "function":
            return self._list_functions(kwargs.get("catalog_name"), kwargs.get("schema_name"))
        else:
            raise ValueError(f"Unsupported asset type: {asset_type}")

    def _list_catalogs(self) -> List[Dict[str, Any]]:
        catalogs = []
        for catalog in self.workspace_client.catalogs.list():
            catalogs.append({
                "name": catalog.name,
                "id": catalog.name,  # Databricks uses name as unique identifier
                "type": "catalog",   # Add type for OpenCITE schema compliance
                "owner": catalog.owner,
                "comment": catalog.comment
            })
        return catalogs

    def _list_schemas(self, catalog_name: str) -> List[Dict[str, Any]]:
        if not catalog_name:
            raise ValueError("catalog_name is required for listing schemas")
        schemas = []
        for schema in self.workspace_client.schemas.list(catalog_name=catalog_name):
            schemas.append({
                "name": schema.name,
                "catalog": schema.catalog_name,  # Normalize to 'catalog' for OpenCITE schema
                "owner": schema.owner,
                "comment": schema.comment
            })
        return schemas

    def _list_tables(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing tables")
        tables = []
        for table in self.workspace_client.tables.list(catalog_name=catalog_name, schema_name=schema_name):
            tables.append({
                "name": table.name,
                "catalog": table.catalog_name,  # Normalize to 'catalog' for OpenCITE schema
                "schema": table.schema_name,    # Normalize to 'schema' for OpenCITE schema
                "table_type": table.table_type.value if table.table_type else None,
                "owner": table.owner,
                "comment": table.comment
            })
        return tables

    def _list_volumes(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing volumes")
        volumes = []
        for volume in self.workspace_client.volumes.list(catalog_name=catalog_name, schema_name=schema_name):
            volumes.append({
                "name": volume.name,
                "catalog": volume.catalog_name,  # Normalize to 'catalog' for OpenCITE schema
                "schema": volume.schema_name,    # Normalize to 'schema' for OpenCITE schema
                "volume_type": volume.volume_type.value if volume.volume_type else None,
                "owner": volume.owner,
                "comment": volume.comment
            })
        return volumes

    def _list_models(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing models")
        models = []
        for model in self.workspace_client.registered_models.list(catalog_name=catalog_name, schema_name=schema_name):
            models.append({
                "name": model.name,
                "catalog": model.catalog_name,  # Normalize to 'catalog' for OpenCITE schema
                "schema": model.schema_name,    # Normalize to 'schema' for OpenCITE schema
                "owner": model.owner,
                "comment": model.comment
            })
        return models

    def _list_functions(self, catalog_name: str, schema_name: str) -> List[Dict[str, Any]]:
        if not catalog_name or not schema_name:
            raise ValueError("catalog_name and schema_name are required for listing functions")
        functions = []
        for func in self.workspace_client.functions.list(catalog_name=catalog_name, schema_name=schema_name):
            functions.append({
                "name": func.name,
                "catalog": func.catalog_name,  # Normalize to 'catalog' for OpenCITE schema
                "schema": func.schema_name,    # Normalize to 'schema' for OpenCITE schema
                "owner": func.owner,
                "comment": func.comment,
                "routine_definition": func.routine_definition
            })
        return functions

    def get_table_usage_from_audit_logs(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Query Databricks system tables to discover which tables have been accessed.

        This queries system.access.audit for table access events to determine
        which tables are actually being used (particularly by AI/ML workloads).

        Args:
            days: Number of days to look back for audit logs (default: 30)

        Returns:
            Dict mapping table full names to usage information:
            {
                "catalog.schema.table": {
                    "access_count": int,
                    "users": [list of users],
                    "notebooks": [list of notebooks],
                    "first_seen": str,
                    "last_seen": str
                }
            }
        """
        table_usage = {}

        try:
            # Query system.access.audit for table access events
            # This requires access to system tables (typically available in Databricks workspaces)
            query = f"""
            SELECT
                request_params.table_full_name as table_name,
                user_identity.email as user,
                request_params.notebook_id as notebook,
                MIN(event_date) as first_access,
                MAX(event_date) as last_access,
                COUNT(*) as access_count
            FROM system.access.audit
            WHERE
                event_date >= DATE_SUB(CURRENT_DATE(), {days})
                AND action_name IN ('getTable', 'readTable', 'scanTable')
                AND request_params.table_full_name IS NOT NULL
            GROUP BY
                request_params.table_full_name,
                user_identity.email,
                request_params.notebook_id
            """

            # Execute query using Databricks SQL
            from databricks import sql

            with sql.connect(
                server_hostname=self.host.replace("https://", ""),
                http_path="/sql/1.0/warehouses/...",  # Would need actual warehouse path
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)

                    for row in cursor.fetchall():
                        table_name = row[0]
                        user = row[1]
                        notebook = row[2]
                        first_access = row[3]
                        last_access = row[4]
                        access_count = row[5]

                        if table_name not in table_usage:
                            table_usage[table_name] = {
                                "access_count": 0,
                                "users": set(),
                                "notebooks": set(),
                                "first_seen": None,
                                "last_seen": None
                            }

                        table_usage[table_name]["access_count"] += access_count
                        table_usage[table_name]["users"].add(user)
                        if notebook:
                            table_usage[table_name]["notebooks"].add(notebook)

                        if not table_usage[table_name]["first_seen"] or first_access < table_usage[table_name]["first_seen"]:
                            table_usage[table_name]["first_seen"] = str(first_access)

                        if not table_usage[table_name]["last_seen"] or last_access > table_usage[table_name]["last_seen"]:
                            table_usage[table_name]["last_seen"] = str(last_access)

        except Exception as e:
            logger.warning(f"Could not query audit logs (may not have access): {e}")
            logger.info("Falling back to trace-based discovery only")

        # Convert sets to lists for JSON serialization
        for table_name in table_usage:
            table_usage[table_name]["users"] = list(table_usage[table_name]["users"])
            table_usage[table_name]["notebooks"] = list(table_usage[table_name]["notebooks"])

        return table_usage

    def get_tables_used_by_ai_workloads(self, days: int = 30, warehouse_id: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get tables that are used by AI/ML workloads specifically.

        This queries system tables to find tables accessed by:
        1. MLflow experiments/runs
        2. Model serving endpoints
        3. Notebooks running AI/ML workloads

        Args:
            days: Number of days to look back
            warehouse_id: SQL warehouse ID for executing queries (optional, will auto-discover if not provided)

        Returns:
            Dict mapping table full names to AI usage information:
            {
                "catalog.schema.table": {
                    "access_count": int,
                    "ai_users": [list of users],
                    "ai_experiments": [list of MLflow experiment IDs],
                    "ai_models": [list of model names],
                    "first_seen": str,
                    "last_seen": str
                }
            }
        """
        ai_table_usage = {}

        try:
            from databricks import sql

            # Get warehouse ID (use provided, configured, or auto-discover)
            wh_id = warehouse_id or self._get_warehouse_id()

            if not wh_id:
                logger.warning("No SQL warehouse available. Cannot query AI workload table usage.")
                logger.info("Please set DATABRICKS_WAREHOUSE_ID or ensure a SQL warehouse exists in your workspace.")
                return ai_table_usage

            http_path = f"/sql/1.0/warehouses/{wh_id}"

            # Query for tables used in MLflow experiments and runs
            query = f"""
            WITH mlflow_table_access AS (
                -- Tables accessed during MLflow experiment runs
                SELECT DISTINCT
                    audit.request_params.table_full_name as table_name,
                    audit.user_identity.email as user,
                    runs.experiment_id,
                    experiments.name as experiment_name,
                    runs.run_id,
                    MIN(audit.event_date) as first_access,
                    MAX(audit.event_date) as last_access,
                    COUNT(*) as access_count
                FROM system.access.audit AS audit
                INNER JOIN (
                    SELECT run_id, experiment_id, artifact_uri
                    FROM system.mlflow.runs
                    WHERE status = 'FINISHED'
                ) AS runs ON audit.request_params.notebook_id LIKE CONCAT('%', runs.run_id, '%')
                INNER JOIN (
                    SELECT experiment_id, name
                    FROM system.mlflow.experiments
                ) AS experiments ON runs.experiment_id = experiments.experiment_id
                WHERE
                    audit.event_date >= DATE_SUB(CURRENT_DATE(), {days})
                    AND audit.action_name IN ('getTable', 'readTable', 'scanTable')
                    AND audit.request_params.table_full_name IS NOT NULL
                GROUP BY
                    audit.request_params.table_full_name,
                    audit.user_identity.email,
                    runs.experiment_id,
                    experiments.name,
                    runs.run_id
            )
            SELECT
                table_name,
                user,
                experiment_id,
                experiment_name,
                MIN(first_access) as first_access,
                MAX(last_access) as last_access,
                SUM(access_count) as total_access_count
            FROM mlflow_table_access
            GROUP BY table_name, user, experiment_id, experiment_name
            ORDER BY table_name, total_access_count DESC
            """

            with sql.connect(
                server_hostname=self.host.replace("https://", ""),
                http_path=http_path,
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)

                    for row in cursor.fetchall():
                        table_name = row[0]
                        user = row[1]
                        experiment_id = row[2]
                        experiment_name = row[3]
                        first_access = row[4]
                        last_access = row[5]
                        access_count = row[6]

                        if table_name not in ai_table_usage:
                            ai_table_usage[table_name] = {
                                "access_count": 0,
                                "ai_users": set(),
                                "ai_experiments": set(),
                                "ai_experiment_names": set(),
                                "first_seen": None,
                                "last_seen": None
                            }

                        ai_table_usage[table_name]["access_count"] += access_count
                        ai_table_usage[table_name]["ai_users"].add(user)
                        ai_table_usage[table_name]["ai_experiments"].add(experiment_id)
                        ai_table_usage[table_name]["ai_experiment_names"].add(experiment_name)

                        if not ai_table_usage[table_name]["first_seen"] or first_access < ai_table_usage[table_name]["first_seen"]:
                            ai_table_usage[table_name]["first_seen"] = str(first_access)

                        if not ai_table_usage[table_name]["last_seen"] or last_access > ai_table_usage[table_name]["last_seen"]:
                            ai_table_usage[table_name]["last_seen"] = str(last_access)

            logger.info(f"Found {len(ai_table_usage)} tables accessed by AI/ML workloads (MLflow experiments)")

        except ImportError:
            logger.warning("databricks-sql-connector not installed. Cannot query AI workload table usage.")
        except Exception as e:
            logger.warning(f"Could not query AI workload table usage: {e}")
            logger.info("This requires SQL warehouse access and system.mlflow tables")

        # Convert sets to lists for JSON serialization
        for table_name in ai_table_usage:
            ai_table_usage[table_name]["ai_users"] = list(ai_table_usage[table_name]["ai_users"])
            ai_table_usage[table_name]["ai_experiments"] = list(ai_table_usage[table_name]["ai_experiments"])
            ai_table_usage[table_name]["ai_experiment_names"] = list(ai_table_usage[table_name]["ai_experiment_names"])

        return ai_table_usage

    def get_tables_used_by_genie(self, days: int = 30, warehouse_id: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Get tables that are used by Databricks Genie spaces.

        This queries system tables to find tables accessed by queries originating from Genie.
        It joins system.query.history with system.access.table_lineage.

        Args:
            days: Number of days to look back
            warehouse_id: SQL warehouse ID for executing queries (optional)

        Returns:
            Dict mapping table full names to Genie usage information:
            {
                "catalog.schema.table": {
                    "access_count": int,
                    "genie_users": [list of users],
                    "genie_spaces": [list of space IDs],
                    "first_seen": str,
                    "last_seen": str
                }
            }
        """
        genie_table_usage = {}

        try:
            from databricks import sql

            wh_id = warehouse_id or self._get_warehouse_id()
            if not wh_id:
                logger.warning("No SQL warehouse available. Cannot query Genie table usage.")
                return genie_table_usage

            http_path = f"/sql/1.0/warehouses/{wh_id}"

            # Query for tables touched by Genie-originated queries
            query = f"""
            SELECT
                lineage.source_table_full_name as table_name,
                history.executed_by as user,
                history.query_source.genie_space_id as space_id,
                MIN(history.start_time) as first_access,
                MAX(history.start_time) as last_access,
                COUNT(*) as access_count
            FROM system.access.table_lineage lineage
            JOIN system.query.history history ON lineage.entity_id = history.statement_id
            WHERE
                lineage.entity_type = 'QUERY'
                AND history.query_source.genie_space_id IS NOT NULL
                AND history.start_time >= DATE_SUB(CURRENT_DATE(), {days})
            GROUP BY
                lineage.source_table_full_name,
                history.executed_by,
                history.query_source.genie_space_id
            """

            with sql.connect(
                server_hostname=self.host.replace("https://", ""),
                http_path=http_path,
                access_token=self.token
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)

                    for row in cursor.fetchall():
                        table_name = row[0]
                        user = row[1]
                        space_id = row[2]
                        first_access = row[3]
                        last_access = row[4]
                        access_count = row[5]

                        if table_name not in genie_table_usage:
                            genie_table_usage[table_name] = {
                                "access_count": 0,
                                "genie_users": set(),
                                "genie_spaces": set(),
                                "first_seen": None,
                                "last_seen": None
                            }

                        genie_table_usage[table_name]["access_count"] += access_count
                        genie_table_usage[table_name]["genie_users"].add(user)
                        genie_table_usage[table_name]["genie_spaces"].add(space_id)

                        if not genie_table_usage[table_name]["first_seen"] or first_access < genie_table_usage[table_name]["first_seen"]:
                            genie_table_usage[table_name]["first_seen"] = str(first_access)

                        if not genie_table_usage[table_name]["last_seen"] or last_access > genie_table_usage[table_name]["last_seen"]:
                            genie_table_usage[table_name]["last_seen"] = str(last_access)

            logger.info(f"Found {len(genie_table_usage)} tables accessed by Genie spaces")

        except ImportError:
            logger.warning("databricks-sql-connector not installed. Cannot query Genie usage.")
        except Exception as e:
            logger.warning(f"Could not query Genie table usage: {e}")
            logger.info("This requires SQL warehouse access and system tables (query.history, access.table_lineage)")

        # Convert sets to lists
        for table_name in genie_table_usage:
            genie_table_usage[table_name]["genie_users"] = list(genie_table_usage[table_name]["genie_users"])
            genie_table_usage[table_name]["genie_spaces"] = list(genie_table_usage[table_name]["genie_spaces"])

        return genie_table_usage
