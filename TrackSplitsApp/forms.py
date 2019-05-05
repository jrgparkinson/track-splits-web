from django import forms
from .validators import validate_gpx_file_extension

class UploadFileForm(forms.Form):
    file = forms.FileField(validators=[validate_gpx_file_extension], label='')