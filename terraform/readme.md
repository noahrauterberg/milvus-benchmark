# Terraform Infrastructure

This Terraform configuration provisions the necessary infrastructure on Google Cloud Platform (GCP) to benchmark perform the Milvus Benchmark.
It automates the deployment of VM instances, disk attachments, monitoring, and network firewall rules.

---

## 2. Requirements
- **Tools:**
  - [Terraform](https://www.terraform.io/downloads.html) `1.5.7` or newer.
  - [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) authenticated to a valid GCP project.
- **Permissions:**
  - The ability to create compute instances, disks, and firewall rules in the target GCP project.

---

## 3. File/Directory Structure Overview

The Terraform configuration includes the following files:

- **`load-generator.tf`:** Provisions a load generator instance with:
  - A persistent disk.
  - Inline startup script to install dependencies, monitoring, and the benchmark application.
- **`milvus.tf`:** Sets up a Milvus database VM with SSD storage and Docker support.
- **`outputs.tf`:** Provides the external and internal IP addresses of the VM instances.
- **`network.tf`:** Configures firewall rules to allow communication between the VMs.
- **`provider.tf`:** Sets the Google Cloud provider and project configurations.
- **`variables.tf`:** Defines input variables for project-specific deployment settings.
- **`terraform.tf`:** Specifies required Terraform and provider versions.

---

## 4. Configuration and Variables

The infrastructure supports the following input variables:

- **`project_id`** *(required)*: The GCP project where resources will be created.
- **`region`** *(required)*: GCP region for deploying resources (e.g., `us-central1`).
- **`zone`** *(required)*: GCP zone for deploying resources (e.g., `us-central1-a`).
- **`load_generator_cidr`** *(optional)*: The IP address range for the load generator. Default is `0.0.0.0/0`.

Define these variables in a `variables.tfvars` file like so:

```hcl
project_id = "your-gcp-project-id"
region     = "us-central1"
zone       = "us-central1-a"
```

---

## 5. Deploying Resources

### Steps to Deploy:
1. **Initialize Terraform:**
   ```bash
   terraform init
   ```

2. **Validate Configuration:**
   ```bash
   terraform validate
   ```

3. **Plan Deployment:**
   ```bash
   terraform plan -var-file="variables.tfvars"
   ```

4. **Apply Deployment:**
   ```bash
   terraform apply -auto-approve -var-file="variables.tfvars"
   ```

---

## 6. Destroying Infrastructure

To clean up all resources created by this configuration:

```bash
terraform destroy -auto-approve -var-file="variables.tfvars"
```

---

## 7. Outputs

The following output variables are provided after successful deployment:

- **`milvus_internal_ip`:** Internal IP of the Milvus instance.
- **`milvus_external_ip`:** External IP of the Milvus instance.
- **`load_generator_internal_ip`:** Internal IP of the load generator instance.
- **`load_generator_external_ip`:** External IP of the load generator instance.

---

## 8. Best Practices

### State Management
- Use a remote backend to store state files securely (e.g., a GCS bucket).

### Avoid Hardcoding
- Leverage `variables.tf` to parameterize configurations and avoid hardcoding project-specific details.

### Security
- Ensure sensitive information (e.g., state files) is not committed to source control.
- If possible, restrict the `load_generator_cidr` to a more secure range than `0.0.0.0/0`.

---

This Terraform configuration simplifies the setup of benchmarking infrastructure for Milvus. Follow the best practices for secure and reliable deployment.
