/* eslint-disable @typescript-eslint/no-explicit-any */
import { jest, test, expect, describe } from "@jest/globals";
import { FakeEmbeddings } from "@langchain/core/utils/testing";

import {
  RedisVectorStore,
  TagFilter,
  NumericFilter,
  TextFilter,
  GeoFilter,
  TimestampFilter,
  Tag,
  Num,
  Text,
  Geo,
  Timestamp,
} from "../vectorstores.js";

const createRedisClientMockup = () => {
  const hSetMock = jest.fn();
  const expireMock = jest.fn();
  const delMock = jest.fn<any>().mockResolvedValue(1);

  return {
    ft: {
      info: jest.fn<any>().mockResolvedValue({
        numDocs: 0,
      }),
      create: jest.fn(),
      search: jest.fn<any>().mockResolvedValue({
        total: 0,
        documents: [],
      }),
      dropIndex: jest.fn(),
    },
    hSet: hSetMock,
    expire: expireMock,
    del: delMock,
    multi: jest.fn<any>().mockImplementation(() => ({
      exec: jest.fn(),
      hSet: hSetMock,
      expire: expireMock,
    })),
  };
};

test("RedisVectorStore with external keys", async () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();

  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
  });

  expect(store).toBeDefined();

  await store.addDocuments(
    [
      {
        pageContent: "hello",
        metadata: {
          a: 1,
          b: { nested: [1, { a: 4 }] },
        },
      },
    ],
    { keys: ["id1"] }
  );

  expect(client.hSet).toHaveBeenCalledTimes(1);
  expect(client.hSet).toHaveBeenCalledWith("id1", {
    content_vector: Buffer.from(new Float32Array([0.1, 0.2, 0.3, 0.4]).buffer),
    content: "hello",
    metadata: `{\\"a\\"\\:1,\\"b\\"\\:{\\"nested\\"\\:[1,{\\"a\\"\\:4}]}}`,
  });

  const results = await store.similaritySearch("goodbye", 1);

  expect(results).toHaveLength(0);
});

test("RedisVectorStore with generated keys", async () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();

  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
  });

  expect(store).toBeDefined();

  await store.addDocuments([{ pageContent: "hello", metadata: { a: 1 } }]);

  expect(client.hSet).toHaveBeenCalledTimes(1);

  const results = await store.similaritySearch("goodbye", 1);

  expect(results).toHaveLength(0);
});

test("RedisVectorStore with TTL", async () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();
  const ttl = 10;
  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
    ttl,
  });

  expect(store).toBeDefined();

  await store.addDocuments([{ pageContent: "hello", metadata: { a: 1 } }]);

  expect(client.hSet).toHaveBeenCalledTimes(1);
  expect(client.expire).toHaveBeenCalledTimes(1);
  expect(client.expire).toHaveBeenCalledWith("doc:documents:0", ttl);
});

test("RedisVectorStore with filters", async () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();

  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
  });

  expect(store).toBeDefined();

  await store.similaritySearch("hello", 1, ["a", "b", "c"]);

  expect(client.ft.search).toHaveBeenCalledWith(
    "documents",
    "@metadata:(a|b|c) => [KNN 1 @content_vector $vector AS vector_score]",
    {
      PARAMS: {
        vector: Buffer.from(new Float32Array([0.1, 0.2, 0.3, 0.4]).buffer),
      },
      RETURN: ["metadata", "content", "vector_score"],
      SORTBY: "vector_score",
      DIALECT: 2,
      LIMIT: {
        from: 0,
        size: 1,
      },
    }
  );
});

test("RedisVectorStore with raw filter", async () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();

  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
  });

  expect(store).toBeDefined();

  await store.similaritySearch("hello", 1, "a b c");

  expect(client.ft.search).toHaveBeenCalledWith(
    "documents",
    "@metadata:(a b c) => [KNN 1 @content_vector $vector AS vector_score]",
    {
      PARAMS: {
        vector: Buffer.from(new Float32Array([0.1, 0.2, 0.3, 0.4]).buffer),
      },
      RETURN: ["metadata", "content", "vector_score"],
      SORTBY: "vector_score",
      DIALECT: 2,
      LIMIT: {
        from: 0,
        size: 1,
      },
    }
  );
});

describe("RedisVectorStore dropIndex", () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();

  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
  });

  test("without deleteDocuments param provided", async () => {
    await store.dropIndex();

    expect(client.ft.dropIndex).toHaveBeenCalledWith("documents", undefined);
  });

  test("with deleteDocuments as false", async () => {
    await store.dropIndex(false);

    expect(client.ft.dropIndex).toHaveBeenCalledWith("documents", undefined);
  });

  test("with deleteDocument as true", async () => {
    await store.dropIndex(true);

    expect(client.ft.dropIndex).toHaveBeenCalledWith("documents", {
      DD: true,
    });
  });

  test("through delete convenience method", async () => {
    await store.delete({ deleteAll: true });

    expect(client.ft.dropIndex).toHaveBeenCalledWith("documents", {
      DD: true,
    });
  });
});

describe("RedisVectorStore createIndex when index does not exist", () => {
  test("calls ft.create with default create options", async () => {
    const client = createRedisClientMockup();
    const embeddings = new FakeEmbeddings();
    const store = new RedisVectorStore(embeddings, {
      redisClient: client as any,
      indexName: "documents",
    });
    store.checkIndexExists = jest.fn<any>().mockResolvedValue(false);

    await store.createIndex();

    expect(client.ft.create).toHaveBeenCalledWith(
      "documents",
      expect.any(Object),
      {
        ON: "HASH",
        PREFIX: "doc:documents:",
      }
    );
  });

  test("calls ft.create with custom options", async () => {
    const client = createRedisClientMockup();
    const embeddings = new FakeEmbeddings();
    const store = new RedisVectorStore(embeddings, {
      redisClient: client as any,
      indexName: "documents",
      createIndexOptions: {
        ON: "JSON",
        FILTER: '@indexName == "documents"',
        SCORE: 0.5,
        MAXTEXTFIELDS: true,
        TEMPORARY: 1000,
        NOOFFSETS: true,
        NOHL: true,
        NOFIELDS: true,
        NOFREQS: true,
        SKIPINITIALSCAN: true,
        STOPWORDS: ["a", "b"],
        LANGUAGE: "German",
      },
    });
    store.checkIndexExists = jest.fn<any>().mockResolvedValue(false);

    await store.createIndex();

    expect(client.ft.create).toHaveBeenCalledWith(
      "documents",
      expect.any(Object),
      {
        ON: "JSON",
        PREFIX: "doc:documents:",
        FILTER: '@indexName == "documents"',
        SCORE: 0.5,
        MAXTEXTFIELDS: true,
        TEMPORARY: 1000,
        NOOFFSETS: true,
        NOHL: true,
        NOFIELDS: true,
        NOFREQS: true,
        SKIPINITIALSCAN: true,
        STOPWORDS: ["a", "b"],
        LANGUAGE: "German",
      }
    );
  });
});

describe("RedisVectorStore delete", () => {
  const client = createRedisClientMockup();
  const embeddings = new FakeEmbeddings();

  const store = new RedisVectorStore(embeddings, {
    redisClient: client as any,
    indexName: "documents",
    keyPrefix: "doc:documents:",
  });

  test("delete documents by ids", async () => {
    const deleteIds = ["doc1", "doc2"];
    await store.delete({ ids: deleteIds });

    expect(client.del).toHaveBeenCalledWith([
      "doc:documents:doc1",
      "doc:documents:doc2",
    ]);
  });

  test("throws error if ids are not provided", async () => {
    await expect(store.delete({ ids: [] })).rejects.toThrow(
      'Invalid parameters passed to "delete".'
    );
  });

  test("throws error if deleteAll is provided as false", async () => {
    await expect(store.delete({ deleteAll: false })).rejects.toThrow(
      'Invalid parameters passed to "delete".'
    );
  });
});

describe("Filter Expression Tests", () => {
  test("TagFilter creates correct query strings", () => {
    const tagFilter = new TagFilter("category", "electronics");
    expect(tagFilter.toString()).toBe("@category:{electronics}");

    const multiTagFilter = new TagFilter("category", ["electronics", "books"]);
    expect(multiTagFilter.toString()).toBe("@category:{electronics|books}");

    const negatedTagFilter = new TagFilter("category", "electronics", true);
    expect(negatedTagFilter.toString()).toBe("(-@category:{electronics})");

    const emptyTagFilter = new TagFilter("category", []);
    expect(emptyTagFilter.toString()).toBe("*");
  });

  test("NumericFilter creates correct query strings", () => {
    const eqFilter = new NumericFilter("price", "eq", 100);
    expect(eqFilter.toString()).toBe("@price:[100 100]");

    const gtFilter = new NumericFilter("price", "gt", 50);
    expect(gtFilter.toString()).toBe("@price:[(50 +inf]");

    const lteFilter = new NumericFilter("price", "lte", 200);
    expect(lteFilter.toString()).toBe("@price:[-inf 200]");

    const betweenFilter = new NumericFilter("price", "between", [50, 200]);
    expect(betweenFilter.toString()).toBe("@price:[50 200]");

    const neFilter = new NumericFilter("price", "ne", 100);
    expect(neFilter.toString()).toBe("(-@price:[100 100])");
  });

  test("TextFilter creates correct query strings", () => {
    const exactFilter = new TextFilter("title", "laptop", "exact");
    expect(exactFilter.toString()).toBe('@title:("laptop")');

    const wildcardFilter = new TextFilter("title", "lap*", "wildcard");
    expect(wildcardFilter.toString()).toBe("@title:(lap*)");

    const fuzzyFilter = new TextFilter("title", "laptop", "fuzzy");
    expect(fuzzyFilter.toString()).toBe("@title:(%%laptop%%)");

    const negatedFilter = new TextFilter("title", "laptop", "exact", true);
    expect(negatedFilter.toString()).toBe('(-@title:("laptop"))');

    const emptyFilter = new TextFilter("title", "", "exact");
    expect(emptyFilter.toString()).toBe("*");
  });

  test("GeoFilter creates correct query strings", () => {
    const geoFilter = new GeoFilter("location", -122.4194, 37.7749, 10, "km");
    expect(geoFilter.toString()).toBe("@location:[-122.4194 37.7749 10 km]");

    const negatedGeoFilter = new GeoFilter(
      "location",
      -122.4194,
      37.7749,
      10,
      "km",
      true
    );
    expect(negatedGeoFilter.toString()).toBe(
      "(-@location:[-122.4194 37.7749 10 km])"
    );
  });

  test("TimestampFilter creates correct query strings", () => {
    const date = new Date("2023-01-01T00:00:00Z");
    const epoch = Math.floor(date.getTime() / 1000);

    const eqFilter = new TimestampFilter("created_at", "eq", date);
    expect(eqFilter.toString()).toBe(`@created_at:[${epoch} ${epoch}]`);

    const gtFilter = new TimestampFilter("created_at", "gt", epoch);
    expect(gtFilter.toString()).toBe(`@created_at:[(${epoch} +inf]`);

    const betweenFilter = new TimestampFilter("created_at", "between", [
      date,
      new Date("2023-12-31T23:59:59Z"),
    ]);
    const endEpoch = Math.floor(
      new Date("2023-12-31T23:59:59Z").getTime() / 1000
    );
    expect(betweenFilter.toString()).toBe(`@created_at:[${epoch} ${endEpoch}]`);
  });

  test("Convenience functions work correctly", () => {
    const tagFilter = Tag("category").eq("electronics");
    expect(tagFilter.toString()).toBe("@category:{electronics}");

    const numFilter = Num("price").between(50, 200);
    expect(numFilter.toString()).toBe("@price:[50 200]");

    const textFilter = Text("title").wildcard("lap*");
    expect(textFilter.toString()).toBe("@title:(lap*)");

    const geoFilter = Geo("location").within(-122.4194, 37.7749, 10, "km");
    expect(geoFilter.toString()).toBe("@location:[-122.4194 37.7749 10 km]");

    const timestampFilter = Timestamp("created_at").gt(new Date("2023-01-01"));
    const epoch = Math.floor(new Date("2023-01-01").getTime() / 1000);
    expect(timestampFilter.toString()).toBe(`@created_at:[(${epoch} +inf]`);
  });

  test("Filter combinations work correctly", () => {
    const tagFilter = Tag("category").eq("electronics");
    const priceFilter = Num("price").between(50, 200);

    const andFilter = tagFilter.and(priceFilter);
    expect(andFilter.toString()).toBe(
      "(@category:{electronics} @price:[50 200])"
    );

    const orFilter = tagFilter.or(priceFilter);
    expect(orFilter.toString()).toBe(
      "(@category:{electronics}|@price:[50 200])"
    );

    // Test wildcard handling in combinations
    const emptyFilter = Tag("empty").eq([]);
    const combinedWithEmpty = tagFilter.and(emptyFilter);
    expect(combinedWithEmpty.toString()).toBe("@category:{electronics}");
  });
});

describe("Metadata Schema Tests", () => {
  test("RedisVectorStore with metadata schema", async () => {
    const client = createRedisClientMockup();
    const embeddings = new FakeEmbeddings();

    const store = new RedisVectorStore(embeddings, {
      redisClient: client as any,
      indexName: "documents",
      metadataSchema: [
        { name: "category", type: "tag" },
        { name: "price", type: "numeric" },
        { name: "title", type: "text" },
        { name: "location", type: "geo" },
        { name: "created_at", type: "timestamp" },
      ],
    });

    expect(store).toBeDefined();
    expect(store.metadataSchema).toHaveLength(5);
  });

  test("Advanced filter with metadata schema", async () => {
    const client = createRedisClientMockup();
    const embeddings = new FakeEmbeddings();

    const store = new RedisVectorStore(embeddings, {
      redisClient: client as any,
      indexName: "documents",
      metadataSchema: [
        { name: "category", type: "tag" },
        { name: "price", type: "numeric" },
      ],
    });

    const complexFilter = Tag("category")
      .eq("electronics")
      .and(Num("price").between(50, 200));

    await store.similaritySearch("test query", 1, complexFilter);

    expect(client.ft.search).toHaveBeenCalledWith(
      "documents",
      "(@category:{electronics} @price:[50 200]) => [KNN 1 @content_vector $vector AS vector_score]",
      expect.objectContaining({
        PARAMS: {
          vector: Buffer.from(new Float32Array([0.1, 0.2, 0.3, 0.4]).buffer),
        },
        RETURN: ["content", "vector_score", "category", "price", "metadata"],
        SORTBY: "vector_score",
        DIALECT: 2,
        LIMIT: {
          from: 0,
          size: 1,
        },
      })
    );
  });

  test("Backward compatibility with legacy filters", async () => {
    const client = createRedisClientMockup();
    const embeddings = new FakeEmbeddings();

    const store = new RedisVectorStore(embeddings, {
      redisClient: client as any,
      indexName: "documents",
    });

    // Test legacy array filter
    await store.similaritySearch("test query", 1, ["electronics", "books"]);

    expect(client.ft.search).toHaveBeenCalledWith(
      "documents",
      "@metadata:(electronics|books) => [KNN 1 @content_vector $vector AS vector_score]",
      expect.objectContaining({
        RETURN: ["metadata", "content", "vector_score"],
      })
    );

    // Test legacy string filter
    await store.similaritySearch("test query", 1, "electronics");

    expect(client.ft.search).toHaveBeenCalledWith(
      "documents",
      "@metadata:(electronics) => [KNN 1 @content_vector $vector AS vector_score]",
      expect.any(Object)
    );
  });
});
