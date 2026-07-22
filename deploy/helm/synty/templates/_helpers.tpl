{{- define "synty.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "synty.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}

{{- define "synty.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "synty.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "synty.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "synty.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- required "serviceAccount.name is required when create=false" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "synty.claimName" -}}
{{- default (include "synty.fullname" .) .Values.persistence.existingClaim -}}
{{- end -}}
