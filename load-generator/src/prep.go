package main

import (
	"context"
	"time"

	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func CreateCollection(
	c *milvusclient.Client,
	ctx context.Context,
	dbName string,
	collection string,
	idFieldName string,
	vecFieldName string,
	dim int,
	fieldName string,
	logger *Logger,
) error {
	/* Create database and schema */
	logger.Log("Creating db...")
	err := c.CreateDatabase(ctx, milvusclient.NewCreateDatabaseOption(dbName))
	if err != nil {
		logger.Log(err.Error())
	}
	err = c.UseDatabase(ctx, milvusclient.NewUseDatabaseOption(dbName))
	if err != nil {
		return err
	}

	logger.Log("Creating Schema...")
	schema := entity.NewSchema().
		WithField(entity.NewField().
			WithName(idFieldName).
			WithIsAutoID(false).
			WithIsPrimaryKey(true).
			WithDataType(entity.FieldTypeInt64),
		).
		WithField(entity.NewField().
			WithName(vecFieldName).
			WithDataType(entity.FieldTypeFloatVector).
			WithDim(int64(dim)),
		).
		WithField(entity.NewField().
			WithName(fieldName).
			WithDataType(entity.FieldTypeVarChar).
			WithMaxLength(128),
		)
	logger.Log("Creating collection...")
	return c.CreateCollection(ctx, milvusclient.NewCreateCollectionOption(collection, schema))
}

func InsertDataset(
	c *milvusclient.Client,
	ctx context.Context,
	collection string,
	idFieldName string,
	vecFieldName string,
	dim int,
	fieldName string,
	data []DataRow,
	batchSize int,
	logger *Logger,
) error {
	logger.Log("Inserting...")
	for start := 0; start < len(data); start += batchSize {
		end := min(start+batchSize, len(data))
		rows := make([]any, 0, batchSize)
		for _, r := range data[start:end] {
			rowMap := map[string]any{
				idFieldName:  r.Id,
				vecFieldName: r.Vector,
				fieldName:    r.Word,
			}
			rows = append(rows, rowMap)
		}
		_, err := c.Insert(ctx, milvusclient.NewRowBasedInsertOption(collection, rows...))
		if err != nil {
			return err
		}
	}
	logger.Log("Insert completed")
	return nil
}

func flushCollection(
	c *milvusclient.Client,
	ctx context.Context,
	collection string,
	logger *Logger,
) error {
	/* Flush and await the flush */
	task, err := c.Flush(ctx, milvusclient.NewFlushOption(collection))
	if err != nil {
		return err
	}
	task.Await(ctx)
	logger.Log("Flush completed")
	return nil
}

func Prepare(
	c *milvusclient.Client,
	dbName string,
	collection string,
	idFieldName string,
	vecFieldName string,
	dim int,
	fieldName string,
	indexParams ConstructionIndexParameters,
	insertBatchSize int,
	datasource DataSource,
) error {
	logger, err := NewLogger("prepare")
	if err != nil {
		return err
	}
	defer logger.Close()

	ctx := context.Background() // we don't want any timeouts for the preparation

	/* Create Database and Collection */
	err = CreateCollection(
		c,
		ctx,
		dbName,
		collection,
		idFieldName,
		vecFieldName,
		dim,
		fieldName,
		logger,
	)
	if err != nil {
		return err
	}

	/* Get Dataset */
	data, err := datasource.GetDataSet()
	if err != nil {
		return err
	}

	/* Persist Data Rows for later recall calculation */
	err = logger.LogDataRows(data)
	if err != nil {
		return err
	}

	/* Insert Dataset */
	InsertDataset(
		c,
		ctx,
		collection,
		idFieldName,
		vecFieldName,
		dim,
		fieldName,
		data,
		insertBatchSize,
		logger,
	)

	/* Create the index */
	indexStartTime := time.Now()

	indexTask, err := c.CreateIndex(ctx, milvusclient.NewCreateIndexOption(
		collection,
		vecFieldName,
		index.NewHNSWIndex(
			index.MetricType(indexParams.distanceMetric),
			indexParams.efConstruction,
			indexParams.M,
		),
	),
	)
	if err != nil {
		return err
	}
	indexTask.Await(ctx)
	indexConstructionTime := time.Since(indexStartTime)
	logger.Logf("Index constructed in %v", indexConstructionTime)

	// Sanity-Check index Creation
	indices, err := c.ListIndexes(ctx, milvusclient.NewListIndexOption(collection))
	if err != nil {
		return err
	}
	logger.Logf("Indices on the collection: %v", indices)

	return nil
}
