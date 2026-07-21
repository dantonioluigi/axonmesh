{{- define "splitflow-cloud.name" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "splitflow-cloud.labels" -}}
app.kubernetes.io/name: splitflow-cloud
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "splitflow-cloud.selectorLabels" -}}
app.kubernetes.io/name: splitflow-cloud
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
