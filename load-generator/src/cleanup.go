package main

import (
	"context"

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

func Cleanup(c *milvusclient.Client, dbName string, collection string) error {
	logger, err := NewLogger("Cleanup")
	if err != nil {
		return err
	}
	defer logger.Close()
	logger.Log("Cleaning up Milvus database and collection...")
	ctx := context.Background()
	err = c.DropCollection(ctx, milvusclient.NewDropCollectionOption(collection))
	if err != nil {
		return err
	}

	logger.Log("Collection dropped successfully")
	err = c.DropDatabase(ctx, milvusclient.NewDropDatabaseOption(dbName))
	if err != nil {
		return err
	}
	logger.Log("Database dropped successfully")
	return nil
}
