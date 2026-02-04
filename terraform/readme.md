# Terraform Infrastructure

This Terraform configuration provisions the necessary infrastructure on the Google Cloud Platform (GCP) to perform the Milvus Benchmark.
It automates the deployment of VM instances, disk attachments, monitoring, and network firewall rules.

## File/Directory Structure Overview

The Terraform configuration includes the following files:

- `load-generator.tf`: Provisions a load generator instance with:
  - A persistent disk.
  - Startup script to install dependencies, monitoring, and the benchmark application.
- `milvus.tf`: Sets up a Milvus instance with:
  - A persistent disk.
  - Startup script to install dependencies, monitoring, and start the Milvus standalone server.
- `offline-recall.tf`: Configures a high compute VM instance for offline recall tasks
- `outputs.tf`: Provides the external and internal IP addresses of the VM instances.
- `network.tf`: Configures firewall rules to allow communication between the VMs.
- `provider.tf`: Sets the Google Cloud provider and project configurations.
- `variables.tf`: Defines input variables for project-specific deployment settings.
- `terraform.tf`: Specifies required Terraform and provider versions.

## Configuration and Variables

The infrastructure supports the following input variables:

- `project_id`: The GCP project where resources will be created.
- `region`: GCP region for deploying resources.
- `zone`: GCP zone for deploying resources.
- `deploy_offline_recall_instance`: Boolean to determine if the offline recall instance should be deployed (defaults to `false`).

Example `variables.tfvars` file:
```hcl
project_id = "gcp-project-id"
region     = "europe-west4"
zone       = "europe-west4a"
```

## Managing Resources

The infrastructure may be managed using standard Terraform commands, i.e.:

```bash
terraform init
terraform plan -var-file=variables.tfvars -out <plan-file>
terraform apply -auto-approve <plan-file>
```

For ease of use, a `Makefile` is provided with the following targets:

- `apply`: Applies the Terraform configuration with auto-approval.
- `destroy`: Destroys all resources created by the configuration (also with auto-approval).

## Outputs

The following output variables are provided after successful deployment:

- `milvus_internal_ip`: Internal IP of the Milvus instance.
- `milvus_external_ip`: External IP of the Milvus instance.
- `load_generator_internal_ip`: Internal IP of the load generator instance.
- `load_generator_external_ip`: External IP of the load generator instance.

## Known Issues and Hints

When deploying the benchmark, we noticed that downloading the GloVe dataset for 200 dimensions is really slow and may at times fail due to network issues.
For that reason, consider removing certain dimensionalities from the startup script in `load-generator.tf` and only download the required dimensionalities.

Further, the `load-generator.tf` startup script does fail when trying to build the load-generator from source due to environment variables.
As of now, this is not fixed and we opted to manually re-run the build commands after SSHing into the load-generator instance.

