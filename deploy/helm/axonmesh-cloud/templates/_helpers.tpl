{{- define "axonmesh-cloud.name" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "axonmesh-cloud.labels" -}}
app.kubernetes.io/name: axonmesh-cloud
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "axonmesh-cloud.selectorLabels" -}}
app.kubernetes.io/name: axonmesh-cloud
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
