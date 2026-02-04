# Load Generator

This subdirectory contains the implementation of the benchmark's load generator, written in Go.
It leverages the Milvus Go SDK to interact with the Milvus server.

## Structure

Unlike typical Go projects, the load-generator is organized as a single `main` package that lives in the `src/` directory.
Therefore, building the load generator requires specifying the source directory:

```bash
go build -o benchmark ./src
```

The file structure tries to follow the main benchmark steps as closely as possible:
```
src
├── cleanup.go # Cleanup Phase (Deleting the Database)
├── collection.go # Collection Phase (Recall calculation)
├── configloader.go # Configuration loading utilities
├── datagenerator.go # Data generation utilities (not in use)
├── datareader.go # Reads in the dataset
├── execution.go # Execution Phase (Running the actual benchmark)
├── jobs_test.go
├── jobs.go # Job generation and scheduling
├── logger.go # Logger utilities (unfortunately a bit mixed-up with data-saving)
├── main.go # The main entry point
├── prep.go # Preparation Phase (Creating the Database, Index, etc.)
├── recall_test.go
├── recall.go # Recall calculation
└── warmup.go # Warmup Phase (Executes 5k queries to warm up the server)
```

## Executing the Load Generator

### Prerequisites

Before running the load generator, ensure the following:

The Milvus server instance is running either on localhost or accessible via network.
When using a remote server, make sure to set the `MILVUS_ADDRESS` environment variable to point to the server's address.
When using the provided infrastructure (`../terraform/`), the internal IP address of the Milvus instance is stored in the `/opt/benchmark/env.sh` file.

Please ensure that the dataset files are available in the specified directory (see `configs/dim-<dimensionality>.txt`).

### Parameters

The load generator accepts the following parameters:
- `<config-id>`: An integer between 1 and 3, representing the configuration Id to be used for the benchmark.
- `<dimensionality>`: An integer representing the dimensionality of the vectors to be used in the benchmark. Supported values are: 50, 100, 200.
- `<optional:perform-recall-calculation>`: An optional boolean flag (`true` or `false`) indicating whether to perform recall calculation directly after the benchmark. This may be useful to save costs when running configurations where the index construction takes a long time. Defaults to `true`.

### Running the Load Generator

After building the load generator, you can execute it with the following command:

```bash
./benchmark <config-id> <dimensionality> <optional:perform-recall-calculation>
```

Note that when running the load-generator in the `/opt/benchmark/` directory, you need root privileges.
To preserve the environment variables, use the `-E` flag with `sudo`:

```bash
sudo -E ./benchmark <config-id> <dimensionality> <optional:perform-recall-calculation>
```

