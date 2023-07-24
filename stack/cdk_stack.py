import aws_cdk
from aws_cdk import (
    Duration,
    Stack, CfnOutput, RemovalPolicy, NestedStack,
)
from aws_cdk import (
    aws_iam as iam,
    aws_kms as kms,
    aws_ec2 as ec2,
    aws_sagemaker as sagemaker,
    aws_s3 as s3,
    aws_logs as logs,
    aws_lambda as _lambda,
    aws_glue as glue,
    aws_s3_deployment as s3_deployment
)
from constructs import Construct


class VpcStack(NestedStack):
    def __init__(self, scope) -> None:
        super().__init__(scope, "vpc-stack")

        self.vpc = self._create_vpc()

        self.sg = ec2.SecurityGroup(
            self,
            "vpce-sg",
            vpc=self.vpc,
            allow_all_outbound=True,
            description="allow tls for vpc endpoint",
        )

        self.vpc.add_interface_endpoint(
            "SMRuntimeEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
            security_groups=[self.sg],
        )
        self.api_endpoint = self.vpc.add_interface_endpoint(
            "APIGatewayEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.APIGATEWAY,
            private_dns_enabled=True,
            security_groups=[self.sg],
        )

    def get_security_groups(self):
        return [self.sg.security_group_id, self.vpc.vpc_default_security_group]

    def get_default_security_groups(self):
        return [self.vpc.vpc_default_security_group]

    def get_private_subnet_ids(self):
        subnets = self.vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        )

        return subnets.subnet_ids

    def get_public_subnets(self):
        return ec2.SubnetSelection(
            subnets=self.vpc.select_subnets(subnet_type=ec2.SubnetType.PUBLIC).subnets
        )

    def get_vpc(self):
        return self.vpc

    def get_api_endpioint(self):
        return self.api_endpoint.vpc_endpoint_id

    def _create_vpc(self) -> ec2.Vpc:
        return ec2.Vpc(
            self,
            "VPC",
            max_azs=4,
            cidr="10.0.0.0/16",
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    name="Public",
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name="Private",
                    cidr_mask=24,
                ),
            ],
            nat_gateways=1,
        )


class WorkshopStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.prefix = "genai-text-to-sql-workshop"

        self.vpc_stack = VpcStack(self)

        s3_bucket = self._create_data_bucket()

        self.test_image_deployment = s3_deployment.BucketDeployment(
            self,
            "s3_deploy_sample_data",
            sources=[s3_deployment.Source.asset("samples/")],
            destination_bucket=s3_bucket,
            destination_key_prefix="samples",
        )

        self._prepare_athena_data(s3_bucket)

        sagemaker_role = self._create_notebook_role(s3_bucket)

        self.create_sagemaker_notebook(sagemaker_role.role_arn)

        self.langchain_layer = self._prepare_lambda_langchain_layer()

        self._create_custom_langchain_function(s3_bucket)

        self._create_langchain_function(s3_bucket)

    def _prepare_athena_data(self, s3_bucket):
        glue_db_name = aws_cdk.CfnParameter(
            self,
            f"{self.prefix}DbName",
            type="String",
            description="Demo Database for GenAI text-to-sql workshop",
            allowed_pattern="[\w-]+",
            default=self.prefix,
        )

        glue_table_name = aws_cdk.CfnParameter(
            self,
            f"{self.prefix}TableName",
            type="String",
            description="Demo table for GenAI text-to-sql workshop",
            allowed_pattern="[\w-]+",
            default=f"sales",
        )

        self.glue_db_name_str = glue_db_name.value_as_string

        glue_database = glue.CfnDatabase(
            self,
            id=self.prefix,
            catalog_id=aws_cdk.Aws.ACCOUNT_ID,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                description=f"Demo device Glue database",
                name=glue_db_name.value_as_string,
            ),
        )

        rawdata_table = glue.CfnTable(
            self,
            f"sales-table",
            catalog_id=aws_cdk.Aws.ACCOUNT_ID,
            database_name=glue_db_name.value_as_string,
            table_input=glue.CfnTable.TableInputProperty(
                name=glue_table_name.value_as_string,
                description="sample sales data",
                parameters={"classification": "csv", },
                table_type='EXTERNAL_TABLE',
                storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                    location="s3://"
                             + s3_bucket.bucket_name
                             + "/samples/data/",

                    input_format="org.apache.hadoop.mapred.TextInputFormat",
                    output_format="org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                    compressed=False,
                    serde_info=glue.CfnTable.SerdeInfoProperty(
                        serialization_library="org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                        parameters={
                            "field.delim": ","
                        }
                    ),
                    columns=[
                        glue.CfnTable.ColumnProperty(name="transaction_date", type="date", comment="Transaction date"),
                        glue.CfnTable.ColumnProperty(
                            name="user_id", type="string", comment="The user who make the purchase",
                        ),
                        glue.CfnTable.ColumnProperty(
                            name="product", type="string", comment="product name. e.g. 'Fruits', 'Ice scream', 'Milk'"
                        ),
                        glue.CfnTable.ColumnProperty(
                            name="price", type="double", comment="The price of the product"
                        ),
                    ],
                ),
            ),
        )

    def _prepare_lambda_langchain_layer(self):
        return _lambda.LayerVersion(
            self,
            'GenaiWorkshopLangchainLayer',
            compatible_runtimes=[_lambda.Runtime.PYTHON_3_10],
            code=_lambda.Code.from_asset("resources/lambda_layer/langchain_layer.zip"),
            layer_version_name="langchain_layer",

        )

    def _create_langchain_function(self, s3_bucket):
        lambda_function_playground = _lambda.Function(
            self,
            "PlayGroundLambdaFn",
            runtime=_lambda.Runtime.PYTHON_3_10,
            allow_public_subnet=True,
            code=_lambda.Code.from_asset("resources/lambda/playground/"),
            handler="handler.lambda_handler",
            description="lambda function for a playground",
            memory_size=256,
            retry_attempts=0,
            timeout=Duration.minutes(1),
            layers=[self.langchain_layer],
            # role=lambda_role,
            # vpc=self.vpc_stack.get_vpc(),
            # vpc_subnets=self.vpc_stack.get_public_subnets(),
            log_retention=logs.RetentionDays.THREE_DAYS,
            environment={
                "ATHENA_BUCKET": s3_bucket.bucket_name,
                "ATHENA_DATABASE": self.glue_db_name_str,
                "ATHENA_REGION": self.region
            },
        )
        lambda_function_playground.add_to_role_policy(iam.PolicyStatement(
            resources=[f"*"],
            actions=[
                "athena:StartQueryExecution",
                "athena:StopQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:ListTableMetadata",
                "glue:GetTables",
                "glue:GetTable",
                "athena:GetTableMetadata",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "sagemaker:InvokeEndpoint"
            ],
        ))

        s3_bucket.grant_read_write(lambda_function_playground)

        lambda_function_playground.apply_removal_policy(RemovalPolicy.DESTROY)

    def _create_custom_langchain_function(self, s3_bucket):
        # Defines trigger sns alarm Lambda resource
        custom_lambda_function = _lambda.Function(
            self,
            "CustomLambdaFn",
            runtime=_lambda.Runtime.PYTHON_3_10,
            allow_public_subnet=True,
            code=_lambda.Code.from_asset("resources/lambda/lambda_custom/"),
            handler="handler.lambda_handler",
            description="lambda function use to finish the task on the custom langchain",
            memory_size=256,
            retry_attempts=0,
            timeout=Duration.minutes(15),
            layers=[self.langchain_layer],
            # role=lambda_role,
            # vpc=self.vpc_stack.get_vpc(),
            # vpc_subnets=self.vpc_stack.get_public_subnets(),
            log_retention=logs.RetentionDays.THREE_DAYS,
            environment={
                "ATHENA_BUCKET": s3_bucket.bucket_name,
                "ATHENA_DATABASE": self.glue_db_name_str,
                "ATHENA_REGION": self.region
            },
        )
        custom_lambda_function.add_to_role_policy(iam.PolicyStatement(
            resources=["*"],
            actions=[
                "athena:StartQueryExecution",
                "athena:StopQueryExecution",
                "athena:GetQueryExecution",
                "athena:GetQueryResults",
                "athena:ListTableMetadata",
                "glue:GetTables",
                "glue:GetTable",
                "athena:GetTableMetadata",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "sagemaker:InvokeEndpoint"
            ],
        ))

        s3_bucket.grant_read_write(custom_lambda_function)

        custom_lambda_function.apply_removal_policy(RemovalPolicy.DESTROY)

    def _create_notebook_role(self, s3_bucket):
        # IAM Roles
        name = "Sagemaker"
        notebook_role = iam.Role(
            self,
            f"{name}Role",
            description="Image detection notebook role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy(
                    self,
                    f"{name}Policy",
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                                "s3:GetBucketAcl",
                                "s3:PutObjectAcl",
                                "s3:AbortMultipartUpload",
                            ],
                            resources=[
                                s3_bucket.arn_for_objects("*"),
                                s3_bucket.bucket_arn,
                                f"arn:aws:s3:::*SageMaker*",
                                f"arn:aws:s3:::*Sagemaker*",
                                f"arn:aws:s3:::*sagemaker*",
                            ],
                            effect=iam.Effect.ALLOW,
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["s3:GetObject"],
                            resources=["*"],
                            conditions={
                                "StringEqualsIgnoreCase": {
                                    "s3:ExistingObjectTag/SageMaker": "true"
                                }
                            },
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["iam:PassRole"],
                            resources=["*"],
                            conditions={
                                "StringEquals": {
                                    "iam:PassedToService": "sagemaker.amazonaws.com"
                                }
                            },
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=["s3:GetBucketAcl", "s3:PutObjectAcl"],
                            resources=[
                                f"arn:aws:s3:::*SageMaker*",
                                f"arn:aws:s3:::*Sagemaker*",
                                f"arn:aws:s3:::*sagemaker*",
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=["s3:ListBucket"],
                            resources=[
                                s3_bucket.bucket_arn,
                                f"arn:aws:s3:::sagemaker*",
                            ],
                            effect=iam.Effect.ALLOW,
                        ),
                        iam.PolicyStatement(
                            actions=["s3:CreateBucket"],
                            resources=[
                                f"arn:aws:s3:::*SageMaker*",
                                f"arn:aws:s3:::*Sagemaker*",
                                f"arn:aws:s3:::*sagemaker*",
                            ],
                            effect=iam.Effect.ALLOW,
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "sagemaker:DescribeEndpointConfig",
                                "sagemaker:DescribeModel",
                                "sagemaker:InvokeEndpoint",
                                "sagemaker:ListTags",
                                "sagemaker:DescribeEndpoint",
                                "sagemaker:CreateModel",
                                "sagemaker:CreateEndpointConfig",
                                "sagemaker:CreateEndpoint",
                                "sagemaker:DeleteModel",
                                "sagemaker:DeleteEndpointConfig",
                                "sagemaker:DeleteEndpoint",
                                "sagemaker:CreateTrainingJob",
                                "sagemaker:DescribeTrainingJob",
                                "sagemaker:UpdateEndpoint",
                                "sagemaker:UpdateEndpointWeightsAndCapacities",
                                "autoscaling:*",
                                "ecr:GetAuthorizationToken",
                                "ecr:GetDownloadUrlForLayer",
                                "ecr:BatchGetImage",
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:SetRepositoryPolicy",
                                "ecr:CompleteLayerUpload",
                                "ecr:BatchDeleteImage",
                                "ecr:UploadLayerPart",
                                "ecr:DeleteRepositoryPolicy",
                                "ecr:InitiateLayerUpload",
                                "ecr:DeleteRepository",
                                "ecr:PutImage",
                                "ecr:CreateRepository",
                                "ec2:CreateVpcEndpoint",
                                "ec2:DescribeRouteTables",
                                "cloudwatch:PutMetricData",
                                "cloudwatch:GetMetricData",
                                "cloudwatch:GetMetricStatistics",
                                "cloudwatch:ListMetrics",
                                "logs:CreateLogStream",
                                "logs:PutLogEvents",
                                "logs:GetLogEvents",
                                "logs:CreateLogGroup",
                                "logs:DescribeLogStreams",
                                "iam:ListRoles",
                                "iam:GetRole",
                                "athena:StartQueryExecution",
                                "athena:StopQueryExecution",
                                "athena:GetQueryExecution",
                                "athena:GetQueryResults",
                                "athena:ListTableMetadata",
                                "athena:GetTableMetadata",
                                "sagemaker:AddTags",
                                "glue:GetTables"
                            ],
                            effect=iam.Effect.ALLOW,
                            resources=["*"],
                        ),
                        iam.PolicyStatement(
                            actions=["kms:Decrypt", "kms:Encrypt", "kms:CreateGrant"],
                            effect=iam.Effect.ALLOW,
                            resources=["*"],
                        )
                    ],
                )
            ],
        )
        notebook_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(
            'AmazonAthenaFullAccess'
        ))

        CfnOutput(self, "SagemakerRoleName", value=notebook_role.role_name)

        return notebook_role

    def _create_data_bucket(self):
        data_bucket = s3.Bucket(
            self,
            "data",
            #bucket_name=self.prefix,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )

        data_bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowSSLRequestsOnly",
                effect=iam.Effect.DENY,
                actions=["s3:*"],
                conditions={"Bool": {"aws:SecureTransport": "false"}},
                principals=[iam.AnyPrincipal()],
                resources=[
                    data_bucket.arn_for_objects("*"),
                    data_bucket.bucket_arn,
                ],
            )
        )

        CfnOutput(self, "DataBucketName", value=data_bucket.bucket_name)
        return data_bucket

    def create_sagemaker_notebook(self, notebook_role_arn):
        notebook_instance_name = "WorkshopNotebook"

        sagemaker_key = kms.Key(
            self,
            id=f"{notebook_instance_name}-key",
            description="AWS key for sagemaker disk",
            enable_key_rotation=True,
        )

        sagemaker_jupyter = sagemaker.CfnNotebookInstance(
            self,
            notebook_instance_name,
            instance_type="ml.t3.medium",
            volume_size_in_gb=20,
            kms_key_id=sagemaker_key.key_id,
            notebook_instance_name=notebook_instance_name,
            role_arn=notebook_role_arn,
            platform_identifier="notebook-al2-v1",
            default_code_repository="https://github.com/eunicetsao/genai-talk-to-your-data-with-large-language-model-notebook"
        )

        CfnOutput(
            self,
            "SagemakerNotebookName",
            value=sagemaker_jupyter.notebook_instance_name,
        )

        return sagemaker_jupyter
