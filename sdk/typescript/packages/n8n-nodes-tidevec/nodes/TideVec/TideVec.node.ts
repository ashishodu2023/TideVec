import {
  type IDataObject,
  type IExecuteFunctions,
  type INodeExecutionData,
  type INodeProperties,
  type INodeType,
  type INodeTypeDescription,
  NodeConnectionTypes,
  NodeOperationError,
} from "n8n-workflow";
import { TideVec as TideVecClient } from "tidevec";
import { type Operation, type OperationParams, type Resource, executeOperation } from "./operations";

const collectionField: INodeProperties = {
  displayName: "Collection",
  name: "collection",
  type: "string",
  default: "",
  required: true,
  displayOptions: { hide: { operation: ["list"] } },
};

const show = (resource: Resource, operation: Operation[]) => ({ show: { resource: [resource], operation } });

export class TideVec implements INodeType {
  description: INodeTypeDescription = {
    displayName: "TideVec",
    name: "tideVec",
    icon: "file:tidevec.svg",
    group: ["transform"],
    version: 1,
    subtitle: '={{$parameter["operation"] + ": " + $parameter["resource"]}}',
    description: "Store and search vectors in TideVec, the vector database that knows when facts change",
    defaults: { name: "TideVec" },
    inputs: [NodeConnectionTypes.Main],
    outputs: [NodeConnectionTypes.Main],
    usableAsTool: true,
    credentials: [{ name: "tideVecApi", required: true }],
    properties: [
      {
        displayName: "Resource",
        name: "resource",
        type: "options",
        noDataExpression: true,
        options: [
          { name: "Collection", value: "collection" },
          { name: "Vector", value: "vector" },
        ],
        default: "vector",
      },
      {
        displayName: "Operation",
        name: "operation",
        type: "options",
        noDataExpression: true,
        displayOptions: { show: { resource: ["collection"] } },
        options: [
          { name: "Create", value: "create", action: "Create a collection", description: "Create unless it exists" },
          { name: "Delete", value: "delete", action: "Delete a collection" },
          { name: "Get", value: "get", action: "Get collection stats" },
          { name: "List", value: "list", action: "List collections" },
        ],
        default: "create",
      },
      {
        displayName: "Operation",
        name: "operation",
        type: "options",
        noDataExpression: true,
        displayOptions: { show: { resource: ["vector"] } },
        options: [
          { name: "Upsert", value: "upsert", action: "Upsert a vector" },
          { name: "Search", value: "search", action: "Search similar vectors" },
          { name: "Delete", value: "deleteVectors", action: "Delete vectors by id or filter" },
        ],
        default: "search",
      },
      collectionField,

      // collection.create
      { displayName: "Dimension", name: "dim", type: "number", default: 1536, required: true, displayOptions: show("collection", ["create"]) },
      {
        displayName: "Metric",
        name: "metric",
        type: "options",
        options: [
          { name: "Cosine", value: "cosine" },
          { name: "Dot Product", value: "dot" },
          { name: "Euclidean (L2)", value: "l2" },
        ],
        default: "cosine",
        displayOptions: show("collection", ["create"]),
      },
      {
        displayName: "Half-Life (Days)",
        name: "halfLifeDays",
        type: "number",
        default: 30,
        description: "How fast recency decays in the score",
        displayOptions: show("collection", ["create"]),
      },
      {
        displayName: "Supersession",
        name: "supersession",
        type: "boolean",
        default: true,
        description: "Whether a new near-duplicate (same match field) marks the older vector as superseded",
        displayOptions: show("collection", ["create"]),
      },
      {
        displayName: "Supersede Match Field",
        name: "supersedeMatchField",
        type: "string",
        default: "doc_id",
        description: "Payload field that must be equal for one vector to supersede another (empty = similarity only)",
        displayOptions: { show: { resource: ["collection"], operation: ["create"], supersession: [true] } },
      },
      {
        displayName: "Supersede Threshold",
        name: "supersedeThreshold",
        type: "number",
        typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 2 },
        default: 0.9,
        description: "Vector score ((1 + cos) / 2) at or above which the newer vector supersedes",
        displayOptions: { show: { resource: ["collection"], operation: ["create"], supersession: [true] } },
      },

      // vector.upsert / search
      {
        displayName: "ID",
        name: "id",
        type: "string",
        default: "",
        required: true,
        displayOptions: show("vector", ["upsert"]),
      },
      {
        displayName: "Embedding",
        name: "embedding",
        type: "json",
        default: "={{ $json.embedding }}",
        required: true,
        description: "Array of numbers, e.g. from an embeddings node or API call",
        displayOptions: show("vector", ["upsert", "search"]),
      },
      {
        displayName: "Payload",
        name: "payload",
        type: "json",
        default: "{}",
        description: "Flat JSON object stored with the vector; non-string values are JSON-encoded",
        displayOptions: show("vector", ["upsert"]),
      },
      {
        displayName: "Created At",
        name: "createdAt",
        type: "string",
        default: "",
        description: "ISO date or epoch milliseconds; empty = now",
        displayOptions: show("vector", ["upsert"]),
      },
      { displayName: "Top K", name: "topK", type: "number", default: 5, displayOptions: show("vector", ["search"]) },
      {
        displayName: "Exclude Superseded",
        name: "excludeSuperseded",
        type: "boolean",
        default: true,
        description: "Whether to drop results that a newer version supersedes",
        displayOptions: show("vector", ["search"]),
      },
      {
        displayName: "Options",
        name: "options",
        type: "collection",
        placeholder: "Add Option",
        default: {},
        displayOptions: show("vector", ["search"]),
        options: [
          {
            displayName: "Temporal Blend",
            name: "temporalBlend",
            type: "number",
            typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 2 },
            default: 0.3,
            description: "Weight of recency in the score (0 = pure similarity)",
          },
          {
            displayName: "Superseded Penalty",
            name: "supersededPenalty",
            type: "number",
            typeOptions: { minValue: 0, maxValue: 1, numberPrecision: 2 },
            default: 0,
          },
        ],
      },
      {
        displayName: "Filter",
        name: "filter",
        type: "string",
        default: "",
        placeholder: "doc_id=refunds",
        description: "Payload filter in key=value form",
        displayOptions: show("vector", ["search", "deleteVectors"]),
      },
      {
        displayName: "IDs",
        name: "ids",
        type: "string",
        default: "",
        placeholder: "id1,id2",
        description: "Comma-separated vector IDs",
        displayOptions: show("vector", ["deleteVectors"]),
      },
    ],
  };

  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const creds = await this.getCredentials("tideVecApi");
    const client = new TideVecClient({
      url: String(creds.url || "http://localhost:6399"),
      apiKey: String(creds.apiKey ?? ""),
    });
    const items = this.getInputData();
    const out: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      const resource = this.getNodeParameter("resource", i) as Resource;
      const operation = this.getNodeParameter("operation", i) as Operation;
      const get = <T>(name: string, fallback?: T) => this.getNodeParameter(name, i, fallback) as T;
      try {
        const params: OperationParams = {
          collection: operation === "list" ? undefined : get<string>("collection"),
        };
        if (resource === "collection" && operation === "create") {
          Object.assign(params, {
            dim: get<number>("dim"),
            metric: get<string>("metric"),
            halfLifeDays: get<number>("halfLifeDays"),
            supersession: get<boolean>("supersession"),
            supersedeMatchField: get<string>("supersedeMatchField", ""),
            supersedeThreshold: get<number>("supersedeThreshold", 0.9),
          });
        } else if (resource === "vector" && operation === "upsert") {
          Object.assign(params, {
            id: get<string>("id"),
            embedding: get<unknown>("embedding"),
            payload: get<unknown>("payload", {}),
            createdAt: get<string>("createdAt", ""),
          });
        } else if (resource === "vector" && operation === "search") {
          const options = get<IDataObject>("options", {});
          Object.assign(params, {
            embedding: get<unknown>("embedding"),
            topK: get<number>("topK"),
            excludeSuperseded: get<boolean>("excludeSuperseded"),
            filter: get<string>("filter", ""),
            temporalBlend: options.temporalBlend as number | undefined,
            supersededPenalty: options.supersededPenalty as number | undefined,
          });
        } else if (resource === "vector" && operation === "deleteVectors") {
          Object.assign(params, { ids: get<string>("ids", ""), filter: get<string>("filter", "") });
        }
        const results = await executeOperation(client, resource, operation, params);
        for (const json of results) out.push({ json: json as IDataObject, pairedItem: { item: i } });
      } catch (error) {
        if (this.continueOnFail()) {
          out.push({ json: { error: (error as Error).message }, pairedItem: { item: i } });
          continue;
        }
        throw new NodeOperationError(this.getNode(), error as Error, { itemIndex: i });
      }
    }
    return [out];
  }
}
