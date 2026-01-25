package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

/**
* LoadIndexConfig reads index configuration in the following format:
* M = 30
* efConstruction = 360
 */
func LoadIndexConfig(configID int, config *Config) error {
	filename := fmt.Sprintf("configs/index-%d.txt", configID)
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open config file %s: %w", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid format on line: %s", line)
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		switch key {
		case "M":
			config.indexParameters.M, err = strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("invalid M value in line: %s", line)
			}
		case "efConstruction":
			config.indexParameters.efConstruction, err = strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("invalid efConstruction value in line: %s", line)
			}
		default:
			return fmt.Errorf("unknown parameter in line: %s", line)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading config file: %w", err)
	}

	// Verify that all required fields are set
	if config.indexParameters.M == 0 {
		return fmt.Errorf("missing required parameter: M")
	}
	if config.indexParameters.efConstruction == 0 {
		return fmt.Errorf("missing required parameter: efConstruction")
	}

	return nil
}

/**
* LoadDimConfig reads dimensionality configuration in the following format:
* dim = 50
* dataFile = ./glove/glove-50.txt
 */
func LoadDimConfig(datasetID int, config *Config) error {
	filename := fmt.Sprintf("configs/dim-%d.txt", datasetID)
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open dimensionality config file %s: %w", filename, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return fmt.Errorf("invalid format in line: %s", line)
		}

		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])

		switch key {
		case "dataFile":
			config.dataFile = value
		case "dim":
			config.dim, err = strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("invalid dim value in line: %s", line)
			}
		default:
			return fmt.Errorf("unknown parameter in line: %s", line)
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading dimensionality config file: %w", err)
	}

	// Validate that all required fields are set
	if config.dataFile == "" {
		return fmt.Errorf("missing required parameter: dataFile")
	}
	if config.dim == 0 {
		return fmt.Errorf("missing required parameter: dim")
	}

	return nil
}
