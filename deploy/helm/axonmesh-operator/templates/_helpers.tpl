{{- define "axonmesh-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "axonmesh-operator.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "axonmesh-operator.serviceAccountName" -}}
{{- default (include "axonmesh-operator.fullname" .) .Values.serviceAccount.name -}}
{{- end -}}

{{- define "axonmesh-operator.labels" -}}
app.kubernetes.io/name: {{ include "axonmesh-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "axonmesh-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "axonmesh-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
