import type {
  createCluster,
  createClient,
  RediSearchSchema,
  SearchOptions,
} from "redis";
import { SchemaFieldTypes, VectorAlgorithms } from "redis";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";

// Filter expression classes for advanced metadata filtering
/**
 * Base class for all filter expressions
 */
export abstract class FilterExpression {
  abstract toString(): string;

  /**
   * Combine this filter with another using AND logic
   */
  and(other: FilterExpression): FilterExpression {
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    return new AndFilter(this, other);
  }

  /**
   * Combine this filter with another using OR logic
   */
  or(other: FilterExpression): FilterExpression {
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    return new OrFilter(this, other);
  }
}

/**
 * Logical AND filter
 */
export class AndFilter extends FilterExpression {
  constructor(private left: FilterExpression, private right: FilterExpression) {
    super();
  }

  toString(): string {
    const leftStr = this.left.toString();
    const rightStr = this.right.toString();

    // Handle wildcard cases
    if (leftStr === "*") return rightStr;
    if (rightStr === "*") return leftStr;

    return `(${leftStr} ${rightStr})`;
  }
}

/**
 * Logical OR filter
 */
export class OrFilter extends FilterExpression {
  constructor(private left: FilterExpression, private right: FilterExpression) {
    super();
  }

  toString(): string {
    const leftStr = this.left.toString();
    const rightStr = this.right.toString();

    // Handle wildcard cases
    if (leftStr === "*" || rightStr === "*") return "*";

    return `(${leftStr}|${rightStr})`;
  }
}



/**
 * Tag filter for exact matching on tag fields
 */
export class TagFilter extends FilterExpression {
  constructor(
    private field: string,
    private values: string | string[] | Set<string>,
    private negate: boolean = false
  ) {
    super();
  }

  static create(field: string) {
    return {
      eq: (values: string | string[] | Set<string>) =>
        new TagFilter(field, values, false),
      ne: (values: string | string[] | Set<string>) =>
        new TagFilter(field, values, true),
    };
  }

  toString(): string {
    if (
      !this.values ||
      (Array.isArray(this.values) && this.values.length === 0) ||
      (this.values && typeof this.values === "object" && "size" in this.values && this.values.size === 0)
    ) {
      return "*"; // Return wildcard for empty filters
    }

    let valueStr: string;
    if (typeof this.values === "string") {
      valueStr = this.values;
    } else if (Array.isArray(this.values)) {
      valueStr = this.values.join("|");
    } else {
      valueStr = Array.from(this.values).join("|");
    }

    const filter = `@${this.field}:{${valueStr}}`;
    return this.negate ? `(-${filter})` : filter;
  }
}

/**
 * Numeric filter for range and exact matching on numeric fields
 */
export class NumericFilter extends FilterExpression {
  constructor(
    private field: string,
    private operator: "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "between",
    private value: number | [number, number],
    private negate: boolean = false
  ) {
    super();
  }

  static create(field: string) {
    return {
      eq: (value: number) => new NumericFilter(field, "eq", value),
      ne: (value: number) => new NumericFilter(field, "ne", value),
      gt: (value: number) => new NumericFilter(field, "gt", value),
      gte: (value: number) => new NumericFilter(field, "gte", value),
      lt: (value: number) => new NumericFilter(field, "lt", value),
      lte: (value: number) => new NumericFilter(field, "lte", value),
      between: (min: number, max: number) =>
        new NumericFilter(field, "between", [min, max]),
    };
  }

  toString(): string {
    let rangeStr: string;

    switch (this.operator) {
      case "eq":
        rangeStr = `[${this.value} ${this.value}]`;
        break;
      case "ne":
        return `(-@${this.field}:[${this.value} ${this.value}])`;
      case "gt":
        rangeStr = `[(${this.value} +inf]`;
        break;
      case "gte":
        rangeStr = `[${this.value} +inf]`;
        break;
      case "lt":
        rangeStr = `[-inf (${this.value}]`;
        break;
      case "lte":
        rangeStr = `[-inf ${this.value}]`;
        break;
      case "between":
        if (Array.isArray(this.value)) {
          rangeStr = `[${this.value[0]} ${this.value[1]}]`;
        } else {
          throw new Error("Between operator requires array of two numbers");
        }
        break;
      default:
        throw new Error(`Unknown numeric operator: ${this.operator}`);
    }

    const filter = `@${this.field}:${rangeStr}`;
    return this.negate ? `(-${filter})` : filter;
  }
}

/**
 * Text filter for full-text search on text fields
 */
export class TextFilter extends FilterExpression {
  constructor(
    private field: string,
    private query: string,
    private operator: "match" | "wildcard" | "fuzzy" | "exact" = "exact",
    private negate: boolean = false
  ) {
    super();
  }

  static create(field: string) {
    return {
      eq: (query: string) => new TextFilter(field, query, "exact"),
      ne: (query: string) => new TextFilter(field, query, "exact", true),
      match: (query: string) => new TextFilter(field, query, "match"),
      wildcard: (query: string) => new TextFilter(field, query, "wildcard"),
      fuzzy: (query: string) => new TextFilter(field, query, "fuzzy"),
    };
  }

  toString(): string {
    if (!this.query || this.query.trim() === "") {
      return "*"; // Return wildcard for empty queries
    }

    let queryStr: string;
    switch (this.operator) {
      case "exact":
        queryStr = `"${this.query}"`;
        break;
      case "match":
        queryStr = this.query;
        break;
      case "wildcard":
        queryStr = this.query; // Wildcards should be included in the query string
        break;
      case "fuzzy":
        queryStr = `%%${this.query}%%`;
        break;
      default:
        queryStr = this.query;
    }

    const filter = `@${this.field}:(${queryStr})`;
    return this.negate ? `(-${filter})` : filter;
  }
}

/**
 * Geographic filter for location-based searches
 */
export class GeoFilter extends FilterExpression {
  constructor(
    private field: string,
    private longitude: number,
    private latitude: number,
    private radius: number,
    private unit: "km" | "mi" | "m" | "ft" = "km",
    private negate: boolean = false
  ) {
    super();
  }

  static create(field: string) {
    return {
      within: (
        longitude: number,
        latitude: number,
        radius: number,
        unit: "km" | "mi" | "m" | "ft" = "km"
      ) => new GeoFilter(field, longitude, latitude, radius, unit),
      outside: (
        longitude: number,
        latitude: number,
        radius: number,
        unit: "km" | "mi" | "m" | "ft" = "km"
      ) => new GeoFilter(field, longitude, latitude, radius, unit, true),
    };
  }

  toString(): string {
    const filter = `@${this.field}:[${this.longitude} ${this.latitude} ${this.radius} ${this.unit}]`;
    return this.negate ? `(-${filter})` : filter;
  }
}

/**
 * Timestamp filter for date/time-based searches
 */
export class TimestampFilter extends FilterExpression {
  constructor(
    private field: string,
    private operator: "eq" | "ne" | "gt" | "gte" | "lt" | "lte" | "between",
    private value: Date | number | [Date | number, Date | number],
    private negate: boolean = false
  ) {
    super();
  }

  static create(field: string) {
    return {
      eq: (value: Date | number) => new TimestampFilter(field, "eq", value),
      ne: (value: Date | number) => new TimestampFilter(field, "ne", value),
      gt: (value: Date | number) => new TimestampFilter(field, "gt", value),
      gte: (value: Date | number) => new TimestampFilter(field, "gte", value),
      lt: (value: Date | number) => new TimestampFilter(field, "lt", value),
      lte: (value: Date | number) => new TimestampFilter(field, "lte", value),
      between: (start: Date | number, end: Date | number) =>
        new TimestampFilter(field, "between", [start, end]),
    };
  }

  private toEpoch(value: Date | number): number {
    return typeof value === "object" && value && "getTime" in value
      ? Math.floor(value.getTime() / 1000)
      : value as number;
  }

  toString(): string {
    let rangeStr: string;

    switch (this.operator) {
      case "eq": {
        const eqValue = this.toEpoch(this.value as Date | number);
        rangeStr = `[${eqValue} ${eqValue}]`;
        break;
      }
      case "ne": {
        const neValue = this.toEpoch(this.value as Date | number);
        return `(-@${this.field}:[${neValue} ${neValue}])`;
      }
      case "gt": {
        const gtValue = this.toEpoch(this.value as Date | number);
        rangeStr = `[(${gtValue} +inf]`;
        break;
      }
      case "gte": {
        const gteValue = this.toEpoch(this.value as Date | number);
        rangeStr = `[${gteValue} +inf]`;
        break;
      }
      case "lt": {
        const ltValue = this.toEpoch(this.value as Date | number);
        rangeStr = `[-inf (${ltValue}]`;
        break;
      }
      case "lte": {
        const lteValue = this.toEpoch(this.value as Date | number);
        rangeStr = `[-inf ${lteValue}]`;
        break;
      }
      case "between": {
        if (Array.isArray(this.value)) {
          const startValue = this.toEpoch(this.value[0]);
          const endValue = this.toEpoch(this.value[1]);
          rangeStr = `[${startValue} ${endValue}]`;
        } else {
          throw new Error("Between operator requires array of two values");
        }
        break;
      }
      default:
        throw new Error(`Unknown timestamp operator: ${this.operator}`);
    }

    const filter = `@${this.field}:${rangeStr}`;
    return this.negate ? `(-${filter})` : filter;
  }
}



// Adapated from internal redis types which aren't exported
/**
 * Type for creating a schema vector field. It includes the algorithm,
 * distance metric, and initial capacity.
 */
export type CreateSchemaVectorField<
  T extends VectorAlgorithms,
  A extends Record<string, unknown>
> = {
  ALGORITHM: T;
  DISTANCE_METRIC: "L2" | "IP" | "COSINE";
  INITIAL_CAP?: number;
} & A;
/**
 * Type for creating a flat schema vector field. It extends
 * CreateSchemaVectorField with a block size property.
 */
export type CreateSchemaFlatVectorField = CreateSchemaVectorField<
  VectorAlgorithms.FLAT,
  {
    BLOCK_SIZE?: number;
  }
>;
/**
 * Type for creating a HNSW schema vector field. It extends
 * CreateSchemaVectorField with M, EF_CONSTRUCTION, and EF_RUNTIME
 * properties.
 */
export type CreateSchemaHNSWVectorField = CreateSchemaVectorField<
  VectorAlgorithms.HNSW,
  {
    M?: number;
    EF_CONSTRUCTION?: number;
    EF_RUNTIME?: number;
  }
>;

type CreateIndexOptions = NonNullable<
  Parameters<ReturnType<typeof createClient>["ft"]["create"]>[3]
>;

export type RedisSearchLanguages = `${NonNullable<
  CreateIndexOptions["LANGUAGE"]
>}`;

export type RedisVectorStoreIndexOptions = Omit<
  CreateIndexOptions,
  "LANGUAGE"
> & { LANGUAGE?: RedisSearchLanguages };

/**
 * Metadata field schema definition for proper indexing
 */
export interface MetadataFieldSchema {
  /** Field name in the metadata */
  name: string;
  /** Field type for indexing */
  type: "tag" | "text" | "numeric" | "geo" | "timestamp";
  /** Additional field options */
  options?: {
    /** For tag fields: separator character (default: |) */
    separator?: string;
    /** For text fields: weight for scoring (default: 1.0) */
    weight?: number;
    /** For numeric fields: whether to sort (default: false) */
    sortable?: boolean;
    /** For all fields: whether to store the field value (default: true) */
    noindex?: boolean;
  };
}

/**
 * Interface for the configuration of the RedisVectorStore. It includes
 * the Redis client, index name, index options, key prefix, content key,
 * metadata key, vector key, filter and ttl.
 */
export interface RedisVectorStoreConfig {
  redisClient:
    | ReturnType<typeof createClient>
    | ReturnType<typeof createCluster>;
  indexName: string;
  indexOptions?: CreateSchemaFlatVectorField | CreateSchemaHNSWVectorField;
  createIndexOptions?: Omit<RedisVectorStoreIndexOptions, "PREFIX">; // PREFIX must be set with keyPrefix
  keyPrefix?: string;
  contentKey?: string;
  metadataKey?: string;
  vectorKey?: string;
  filter?: RedisVectorStoreFilterType;
  ttl?: number; // ttl in second
  /**
   * Metadata schema for proper field indexing. When provided, metadata fields
   * will be indexed individually instead of as a single JSON string.
   */
  metadataSchema?: MetadataFieldSchema[];
}

/**
 * Interface for the options when adding documents to the
 * RedisVectorStore. It includes keys and batch size.
 */
export interface RedisAddOptions {
  keys?: string[];
  batchSize?: number;
}

/**
 * Type for the filter used in the RedisVectorStore. Supports multiple formats:
 * - string[]: Array of strings for simple OR filtering (legacy format)
 * - string: Raw Redis query string for custom filters
 * - FilterExpression: Advanced filter expressions with proper typing
 */
export type RedisVectorStoreFilterType = string[] | string | FilterExpression;

/**
 * Class representing a RedisVectorStore. It extends the VectorStore class
 * and includes methods for adding documents and vectors, performing
 * similarity searches, managing the index, and more.
 */
export class RedisVectorStore extends VectorStore {
  declare FilterType: RedisVectorStoreFilterType;

  private redisClient:
    | ReturnType<typeof createClient>
    | ReturnType<typeof createCluster>;

  indexName: string;

  indexOptions: CreateSchemaFlatVectorField | CreateSchemaHNSWVectorField;

  createIndexOptions: CreateIndexOptions;

  keyPrefix: string;

  contentKey: string;

  metadataKey: string;

  vectorKey: string;

  filter?: RedisVectorStoreFilterType;

  ttl?: number;

  metadataSchema?: MetadataFieldSchema[];

  _vectorstoreType(): string {
    return "redis";
  }

  constructor(
    embeddings: EmbeddingsInterface,
    _dbConfig: RedisVectorStoreConfig
  ) {
    super(embeddings, _dbConfig);

    this.redisClient = _dbConfig.redisClient;
    this.indexName = _dbConfig.indexName;
    this.indexOptions = _dbConfig.indexOptions ?? {
      ALGORITHM: VectorAlgorithms.HNSW,
      DISTANCE_METRIC: "COSINE",
    };
    this.keyPrefix = _dbConfig.keyPrefix ?? `doc:${this.indexName}:`;
    this.contentKey = _dbConfig.contentKey ?? "content";
    this.metadataKey = _dbConfig.metadataKey ?? "metadata";
    this.vectorKey = _dbConfig.vectorKey ?? "content_vector";
    this.filter = _dbConfig.filter;
    this.ttl = _dbConfig.ttl;
    this.metadataSchema = _dbConfig.metadataSchema;
    this.createIndexOptions = {
      ON: "HASH",
      PREFIX: this.keyPrefix,
      ...(_dbConfig.createIndexOptions as CreateIndexOptions),
    };
  }

  /**
   * Method for adding documents to the RedisVectorStore. It first converts
   * the documents to texts and then adds them as vectors.
   * @param documents The documents to add.
   * @param options Optional parameters for adding the documents.
   * @returns A promise that resolves when the documents have been added.
   */
  async addDocuments(documents: Document[], options?: RedisAddOptions) {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }

  /**
   * Method for adding vectors to the RedisVectorStore. It checks if the
   * index exists and creates it if it doesn't, then adds the vectors in
   * batches.
   * @param vectors The vectors to add.
   * @param documents The documents associated with the vectors.
   * @param keys Optional keys for the vectors.
   * @param batchSize The size of the batches in which to add the vectors. Defaults to 1000.
   * @returns A promise that resolves when the vectors have been added.
   */
  async addVectors(
    vectors: number[][],
    documents: Document[],
    { keys, batchSize = 1000 }: RedisAddOptions = {}
  ) {
    if (!vectors.length || !vectors[0].length) {
      throw new Error("No vectors provided");
    }
    // check if the index exists and create it if it doesn't
    await this.createIndex(vectors[0].length);

    const info = await this.redisClient.ft.info(this.indexName);
    const lastKeyCount = parseInt(info.numDocs, 10) || 0;
    const multi = this.redisClient.multi();

    vectors.map(async (vector, idx) => {
      const key =
        keys && keys.length
          ? keys[idx]
          : `${this.keyPrefix}${idx + lastKeyCount}`;
      const metadata =
        documents[idx] && documents[idx].metadata
          ? documents[idx].metadata
          : {};

      const hashData: Record<string, string | number | Buffer> = {
        [this.vectorKey]: this.getFloat32Buffer(vector),
        [this.contentKey]: documents[idx].pageContent,
      };

      // Handle metadata based on schema configuration
      if (this.metadataSchema && this.metadataSchema.length > 0) {
        // Store individual metadata fields for proper indexing
        for (const fieldSchema of this.metadataSchema) {
          const fieldValue = metadata[fieldSchema.name];
          if (fieldValue !== undefined && fieldValue !== null) {
            // Convert values based on field type
            switch (fieldSchema.type) {
              case "tag":
                hashData[fieldSchema.name] = Array.isArray(fieldValue)
                  ? fieldValue.join(fieldSchema.options?.separator || "|")
                  : String(fieldValue);
                break;
              case "text":
                hashData[fieldSchema.name] = String(fieldValue);
                break;
              case "numeric":
                hashData[fieldSchema.name] = Number(fieldValue);
                break;
              case "geo":
                // Expect geo values as "longitude,latitude" string or [lon, lat] array
                if (Array.isArray(fieldValue) && fieldValue.length === 2) {
                  hashData[
                    fieldSchema.name
                  ] = `${fieldValue[0]},${fieldValue[1]}`;
                } else {
                  hashData[fieldSchema.name] = String(fieldValue);
                }
                break;
              case "timestamp": {
                // Convert Date objects to epoch timestamps
                if (typeof fieldValue === "object" && fieldValue && "getTime" in fieldValue) {
                  hashData[fieldSchema.name] = Math.floor(
                    (fieldValue as Date).getTime() / 1000
                  );
                } else {
                  hashData[fieldSchema.name] = Number(fieldValue);
                }
                break;
              }
              default:
                hashData[fieldSchema.name] = String(fieldValue);
            }
          }
        }
        // Also store the full metadata as JSON for backward compatibility
        hashData[this.metadataKey] = this.escapeSpecialChars(
          JSON.stringify(metadata)
        );
      } else {
        // Legacy behavior: store metadata as JSON string
        hashData[this.metadataKey] = this.escapeSpecialChars(
          JSON.stringify(metadata)
        );
      }

      multi.hSet(key, hashData);

      if (this.ttl) {
        multi.expire(key, this.ttl);
      }

      // write batch
      if (idx % batchSize === 0) {
        await multi.exec();
      }
    });

    // insert final batch
    await multi.exec();
  }

  /**
   * Method for performing a similarity search in the RedisVectorStore. It
   * returns the documents and their scores.
   * @param query The query vector.
   * @param k The number of nearest neighbors to return.
   * @param filter Optional filter to apply to the search.
   * @returns A promise that resolves to an array of documents and their scores.
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: RedisVectorStoreFilterType
  ): Promise<[Document, number][]> {
    if (filter && this.filter) {
      throw new Error("cannot provide both `filter` and `this.filter`");
    }

    const _filter = filter ?? this.filter;
    const results = await this.redisClient.ft.search(
      this.indexName,
      ...this.buildQuery(query, k, _filter)
    );
    const result: [Document, number][] = [];

    if (results.total) {
      for (const res of results.documents) {
        if (res.value) {
          const document = res.value;
          if (document.vector_score) {
            // Reconstruct metadata from individual fields if schema is configured
            let metadata: Record<string, unknown> = {};

            if (this.metadataSchema && this.metadataSchema.length > 0) {
              // Build metadata from individual schema fields
              for (const fieldSchema of this.metadataSchema) {
                const fieldValue = document[fieldSchema.name];
                if (fieldValue !== undefined && fieldValue !== null) {
                  switch (fieldSchema.type) {
                    case "tag": {
                      // Convert back from pipe-separated string if needed
                      const separator = fieldSchema.options?.separator || "|";
                      if (
                        typeof fieldValue === "string" &&
                        fieldValue.includes(separator)
                      ) {
                        metadata[fieldSchema.name] =
                          fieldValue.split(separator);
                      } else {
                        metadata[fieldSchema.name] = fieldValue;
                      }
                      break;
                    }
                    case "numeric":
                      metadata[fieldSchema.name] = Number(fieldValue);
                      break;
                    case "geo":
                      // Convert back to [longitude, latitude] array if it's a string
                      if (
                        typeof fieldValue === "string" &&
                        fieldValue.includes(",")
                      ) {
                        const [lon, lat] = fieldValue.split(",").map(Number);
                        metadata[fieldSchema.name] = [lon, lat];
                      } else {
                        metadata[fieldSchema.name] = fieldValue;
                      }
                      break;
                    case "timestamp":
                      // Convert epoch back to Date object
                      metadata[fieldSchema.name] = new Date(
                        Number(fieldValue) * 1000
                      );
                      break;
                    default:
                      metadata[fieldSchema.name] = fieldValue;
                  }
                }
              }
            }

            // Also try to parse the JSON metadata field for any additional fields
            try {
              const jsonMetadata = JSON.parse(
                this.unEscapeSpecialChars(
                  (document[this.metadataKey] ?? "{}") as string
                )
              );
              // Merge with schema-based metadata, giving priority to schema fields
              metadata = { ...jsonMetadata, ...metadata };
            } catch (error) {
              // If JSON parsing fails, use only schema-based metadata
              if (!this.metadataSchema || this.metadataSchema.length === 0) {
                metadata = {};
              }
            }

            result.push([
              new Document({
                pageContent: (document[this.contentKey] ?? "") as string,
                metadata,
              }),
              Number(document.vector_score),
            ]);
          }
        }
      }
    }

    return result;
  }

  /**
   * Static method for creating a new instance of RedisVectorStore from
   * texts. It creates documents from the texts and metadata, then adds them
   * to the RedisVectorStore.
   * @param texts The texts to add.
   * @param metadatas The metadata associated with the texts.
   * @param embeddings The embeddings to use.
   * @param dbConfig The configuration for the RedisVectorStore.
   * @param docsOptions The document options to use.
   * @returns A promise that resolves to a new instance of RedisVectorStore.
   */
  static fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    dbConfig: RedisVectorStoreConfig,
    docsOptions?: RedisAddOptions
  ): Promise<RedisVectorStore> {
    const docs: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas;
      const newDoc = new Document({
        pageContent: texts[i],
        metadata,
      });
      docs.push(newDoc);
    }
    return RedisVectorStore.fromDocuments(
      docs,
      embeddings,
      dbConfig,
      docsOptions
    );
  }

  /**
   * Static method for creating a new instance of RedisVectorStore from
   * documents. It adds the documents to the RedisVectorStore.
   * @param docs The documents to add.
   * @param embeddings The embeddings to use.
   * @param dbConfig The configuration for the RedisVectorStore.
   * @param docsOptions The document options to use.
   * @returns A promise that resolves to a new instance of RedisVectorStore.
   */
  static async fromDocuments(
    docs: Document[],
    embeddings: EmbeddingsInterface,
    dbConfig: RedisVectorStoreConfig,
    docsOptions?: RedisAddOptions
  ): Promise<RedisVectorStore> {
    const instance = new this(embeddings, dbConfig);
    await instance.addDocuments(docs, docsOptions);
    return instance;
  }

  /**
   * Method for checking if an index exists in the RedisVectorStore.
   * @returns A promise that resolves to a boolean indicating whether the index exists.
   */
  async checkIndexExists() {
    try {
      await this.redisClient.ft.info(this.indexName);
    } catch (err) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((err as any)?.message.includes("unknown command")) {
        throw new Error(
          "Failed to run FT.INFO command. Please ensure that you are running a RediSearch-capable Redis instance: https://js.langchain.com/docs/integrations/vectorstores/redis/#setup"
        );
      }
      // index doesn't exist
      return false;
    }

    return true;
  }

  /**
   * Method for creating an index in the RedisVectorStore. If the index
   * already exists, it does nothing.
   * @param dimensions The dimensions of the index
   * @returns A promise that resolves when the index has been created.
   */
  async createIndex(dimensions = 1536): Promise<void> {
    if (await this.checkIndexExists()) {
      return;
    }

    const schema: RediSearchSchema = {
      [this.vectorKey]: {
        type: SchemaFieldTypes.VECTOR,
        TYPE: "FLOAT32",
        DIM: dimensions,
        ...this.indexOptions,
      },
      [this.contentKey]: SchemaFieldTypes.TEXT,
      [this.metadataKey]: SchemaFieldTypes.TEXT,
    };

    // Add metadata schema fields if configured
    if (this.metadataSchema && this.metadataSchema.length > 0) {
      for (const fieldSchema of this.metadataSchema) {
        const fieldOptions: Record<string, unknown> = {};

        switch (fieldSchema.type) {
          case "tag":
            schema[fieldSchema.name] = {
              type: SchemaFieldTypes.TAG,
              SEPARATOR: fieldSchema.options?.separator || "|",
              ...fieldOptions,
            };
            break;
          case "text":
            schema[fieldSchema.name] = {
              type: SchemaFieldTypes.TEXT,
              WEIGHT: fieldSchema.options?.weight || 1.0,
              ...fieldOptions,
            };
            break;
          case "numeric":
            schema[fieldSchema.name] = {
              type: SchemaFieldTypes.NUMERIC,
              SORTABLE: fieldSchema.options?.sortable ? true : undefined,
              ...fieldOptions,
            };
            break;
          case "geo":
            schema[fieldSchema.name] = {
              type: SchemaFieldTypes.GEO,
              ...fieldOptions,
            };
            break;
          case "timestamp":
            // Timestamps are stored as numeric epoch values
            schema[fieldSchema.name] = {
              type: SchemaFieldTypes.NUMERIC,
              SORTABLE: fieldSchema.options?.sortable !== false ? true : undefined, // Default to sortable for timestamps
              ...fieldOptions,
            };
            break;
          default:
            // Default to text for unknown types
            schema[fieldSchema.name] = {
              type: SchemaFieldTypes.TEXT,
              ...fieldOptions,
            };
        }

        // Apply common options
        if (fieldSchema.options?.noindex) {
          const schemaField = schema[fieldSchema.name];
          if (typeof schemaField === 'object' && schemaField && 'NOINDEX' in schemaField) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (schemaField as any).NOINDEX = true;
          }
        }
      }
    }

    await this.redisClient.ft.create(
      this.indexName,
      schema,
      this.createIndexOptions
    );
  }

  /**
   * Method for dropping an index from the RedisVectorStore.
   * @param deleteDocuments Optional boolean indicating whether to drop the associated documents.
   * @returns A promise that resolves to a boolean indicating whether the index was dropped.
   */
  async dropIndex(deleteDocuments?: boolean): Promise<boolean> {
    try {
      const options = deleteDocuments ? { DD: deleteDocuments } : undefined;
      await this.redisClient.ft.dropIndex(this.indexName, options);

      return true;
    } catch (err) {
      return false;
    }
  }

  /**
   * Deletes vectors from the vector store.
   *
   * Supports two deletion modes:
   * - Delete all documents by dropping the entire index and recreating it
   * - Delete specific documents by their IDs using Redis DEL operation
   *
   * @param params - The deletion parameters. Must be one of:
   *   - `{ deleteAll: boolean }` - If true, drops the entire index and all associated documents
   *   - `{ ids: string[] }` - Array of document IDs to delete. IDs will be automatically prefixed with the configured keyPrefix
   * @returns A promise that resolves when the deletion operation is complete
   * @throws {Error} Throws an error if invalid parameters are provided (neither deleteAll nor ids specified)
   *
   * @example
   * Delete all documents:
   * ```typescript
   * await vectorStore.delete({ deleteAll: true });
   * ```
   *
   * @example
   * Delete specific documents by ID:
   * ```typescript
   * await vectorStore.delete({ ids: ['doc1', 'doc2', 'doc3'] });
   * ```
   */
  async delete(
    params: { deleteAll: boolean } | { ids: string[] }
  ): Promise<void> {
    if ("deleteAll" in params && params.deleteAll) {
      await this.dropIndex(true);
    } else if ("ids" in params && params.ids && params.ids.length > 0) {
      const keys = params.ids.map((id) => `${this.keyPrefix}${id}`);

      await this.redisClient.del(keys);
    } else {
      throw new Error(`Invalid parameters passed to "delete".`);
    }
  }

  private buildQuery(
    query: number[],
    k: number,
    filter?: RedisVectorStoreFilterType
  ): [string, SearchOptions] {
    const vectorScoreField = "vector_score";

    let hybridFields = "*";
    // if a filter is set, modify the hybrid query
    if (filter) {
      const filterStr = this.prepareFilter(filter);
      if (filterStr && filterStr !== "*") {
        hybridFields = filterStr;
      }
    }

    const baseQuery = `${hybridFields} => [KNN ${k} @${this.vectorKey} $vector AS ${vectorScoreField}]`;

    // Build return fields - include metadata schema fields if configured
    const returnFields = [];

    if (this.metadataSchema && this.metadataSchema.length > 0) {
      // When metadata schema is configured, use the expected order from tests
      returnFields.push(this.contentKey, vectorScoreField);
      // Add individual metadata fields
      for (const fieldSchema of this.metadataSchema) {
        returnFields.push(fieldSchema.name);
      }
      // Add the metadata JSON field last for backward compatibility
      returnFields.push(this.metadataKey);
    } else {
      // Default order for backward compatibility
      returnFields.push(this.metadataKey, this.contentKey, vectorScoreField);
    }

    const options: SearchOptions = {
      PARAMS: {
        vector: this.getFloat32Buffer(query),
      },
      RETURN: returnFields,
      SORTBY: vectorScoreField,
      DIALECT: 2,
      LIMIT: {
        from: 0,
        size: k,
      },
    };

    return [baseQuery, options];
  }

  private prepareFilter(filter: RedisVectorStoreFilterType): string {
    if (!filter) {
      return "*";
    }

    // Check for arrays first (before the general object check)
    if (Array.isArray(filter)) {
      if (filter.length === 0) {
        return "*";
      }
      // Legacy behavior: array of strings for OR filtering in metadata field
      const escapedFilters = filter.map(this.escapeSpecialChars).join("|");
      return `@${this.metadataKey}:(${escapedFilters})`;
    }

    // Check for FilterExpression objects (but not arrays)
    if (typeof filter === "object" && filter && !Array.isArray(filter) && "toString" in filter && typeof filter.toString === "function") {
      // Use the filter expression's toString method
      return filter.toString();
    }

    if (typeof filter === "string") {
      if (filter.trim() === "") {
        return "*";
      }
      // Raw Redis query string - check if it already contains field references
      if (filter.includes("@")) {
        return filter;
      } else {
        // Assume it's a metadata search
        return `@${this.metadataKey}:(${filter})`;
      }
    }

    return "*";
  }

  /**
   * Escapes all '-', ':', and '"' characters.
   * RediSearch considers these all as special characters, so we need
   * to escape them
   * @see https://redis.io/docs/stack/search/reference/query_syntax
   *
   * @param str
   * @returns
   */
  private escapeSpecialChars(str: string) {
    return str
      .replaceAll("-", "\\-")
      .replaceAll(":", "\\:")
      .replaceAll(`"`, `\\"`);
  }

  /**
   * Unescapes all '-', ':', and '"' characters, returning the original string
   *
   * @param str
   * @returns
   */
  private unEscapeSpecialChars(str: string) {
    return str
      .replaceAll("\\-", "-")
      .replaceAll("\\:", ":")
      .replaceAll(`\\"`, `"`);
  }

  /**
   * Converts the vector to the buffer Redis needs to
   * correctly store an embedding
   *
   * @param vector
   * @returns Buffer
   */
  private getFloat32Buffer(vector: number[]) {
    return Buffer.from(new Float32Array(vector).buffer);
  }
}

// Convenience functions for creating filters (similar to Python RedisVL)
/**
 * Create a tag filter for exact matching on tag fields
 * @param field The field name to filter on
 * @returns Object with eq and ne methods for creating tag filters
 */
export function Tag(field: string) {
  return TagFilter.create(field);
}

/**
 * Create a numeric filter for range and exact matching on numeric fields
 * @param field The field name to filter on
 * @returns Object with comparison methods for creating numeric filters
 */
export function Num(field: string) {
  return NumericFilter.create(field);
}

/**
 * Create a text filter for full-text search on text fields
 * @param field The field name to filter on
 * @returns Object with text search methods for creating text filters
 */
export function Text(field: string) {
  return TextFilter.create(field);
}

/**
 * Create a geographic filter for location-based searches
 * @param field The field name to filter on
 * @returns Object with geographic search methods for creating geo filters
 */
export function Geo(field: string) {
  return GeoFilter.create(field);
}

/**
 * Create a timestamp filter for date/time-based searches
 * @param field The field name to filter on
 * @returns Object with timestamp comparison methods for creating timestamp filters
 */
export function Timestamp(field: string) {
  return TimestampFilter.create(field);
}
