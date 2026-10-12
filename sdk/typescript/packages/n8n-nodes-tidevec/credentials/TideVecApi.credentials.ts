import type { IAuthenticateGeneric, ICredentialTestRequest, ICredentialType, INodeProperties } from "n8n-workflow";

export class TideVecApi implements ICredentialType {
  name = "tideVecApi";

  displayName = "TideVec API";

  documentationUrl = "https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript/packages/n8n-nodes-tidevec";

  properties: INodeProperties[] = [
    {
      displayName: "Server URL",
      name: "url",
      type: "string",
      default: "http://localhost:6399",
      placeholder: "http://tidevec:6399",
    },
    {
      displayName: "API Key",
      name: "apiKey",
      type: "string",
      typeOptions: { password: true },
      default: "",
      description: "Sent as X-Api-Key. Leave empty for servers started with --no-require-auth.",
    },
  ];

  authenticate: IAuthenticateGeneric = {
    type: "generic",
    properties: {
      headers: { "X-Api-Key": "={{$credentials.apiKey}}" },
    },
  };

  test: ICredentialTestRequest = {
    request: {
      baseURL: "={{$credentials.url}}",
      url: "/v1/collections",
    },
  };
}
