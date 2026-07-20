{{- define "yolosplit-cloud.name" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "yolosplit-cloud.labels" -}}
app.kubernetes.io/name: yolosplit-cloud
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "yolosplit-cloud.selectorLabels" -}}
app.kubernetes.io/name: yolosplit-cloud
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
