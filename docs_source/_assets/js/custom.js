(function() {
  if (!location.href.endsWith('/')) {
    window.location = location.href + '/'
  }
}())


function download_file(file_link){

		var parts = arguments[0].split('/');
		var filename = parts.pop() || parts.pop();  // handle potential trailing slash
	
		axios({
				url:arguments[0],
				method:'GET',
				responseType: 'blob'
})
.then((response) => {
			 const url = window.URL
			 .createObjectURL(new Blob([response.data]));
							const link = document.createElement('a');
							link.href = url;
							link.setAttribute('download', filename);
							document.body.appendChild(link);
							link.click();
})
}