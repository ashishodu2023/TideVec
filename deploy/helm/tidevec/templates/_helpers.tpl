{{- define "tidevec.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tidevec.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "tidevec.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tidevec.labels" -}}
helm.sh/chart: {{ include "tidevec.chart" . }}
{{ include "tidevec.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "tidevec.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tidevec.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "tidevec.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "tidevec.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "tidevec.headlessName" -}}
{{- printf "%s-headless" (include "tidevec.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "tidevec.image" -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) }}
{{- end }}

{{/* Secret holding the API key (and the cluster secret). */}}
{{- define "tidevec.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "tidevec.fullname" .) }}
{{- end }}
{{- end }}

{{- define "tidevec.secretKey" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecretKey }}
{{- else -}}
api-key
{{- end }}
{{- end }}

{{/* Value from the release's existing chart-managed Secret, so generated keys survive upgrades. */}}
{{- define "tidevec.existingSecretValue" -}}
{{- $s := lookup "v1" "Secret" .ctx.Release.Namespace (printf "%s-secret" (include "tidevec.fullname" .ctx)) -}}
{{- if and $s $s.data (hasKey $s.data .key) -}}
{{- index $s.data .key | b64dec -}}
{{- end -}}
{{- end }}

{{/* Raft peer list: 0=http://<pod-0>.<headless>.<ns>.svc:<port>,1=... */}}
{{- define "tidevec.raftPeers" -}}
{{- $peers := list -}}
{{- $full := include "tidevec.fullname" . -}}
{{- $headless := include "tidevec.headlessName" . -}}
{{- range $i := until (int .Values.replicaCount) -}}
{{- $peers = append $peers (printf "%d=http://%s-%d.%s.%s.svc:%d" $i $full $i $headless $.Release.Namespace 6399) -}}
{{- end -}}
{{- join "," $peers -}}
{{- end }}

{{- define "tidevec.validate" -}}
{{- if and (gt (int .Values.replicaCount) 1) (not .Values.cluster.enabled) -}}
{{- fail "replicaCount > 1 needs cluster.enabled=true: without Raft each pod is an independent database behind one Service" -}}
{{- end -}}
{{- if .Values.cluster.enabled -}}
{{- if or (lt (int .Values.replicaCount) 3) (eq (mod (int .Values.replicaCount) 2) 0) -}}
{{- fail "cluster.enabled=true needs an odd replicaCount >= 3" -}}
{{- end -}}
{{- end -}}
{{- if .Values.tls.enabled -}}
{{- if not .Values.tls.secretName -}}
{{- fail "tls.enabled=true needs tls.secretName (a kubernetes.io/tls Secret)" -}}
{{- end -}}
{{- if .Values.cluster.enabled -}}
{{- fail "tls.enabled=true is not supported with cluster.enabled=true (Raft peers use plain HTTP)" -}}
{{- end -}}
{{- end -}}
{{- if and (not .Values.config.requireAuth) .Values.config.multiTenant -}}
{{- fail "config.multiTenant=true needs config.requireAuth=true" -}}
{{- end -}}
{{- end }}

{{/* http or https, following tls.enabled */}}
{{- define "tidevec.scheme" -}}
{{- ternary "https" "http" .Values.tls.enabled -}}
{{- end }}
