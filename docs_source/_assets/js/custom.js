(function() {
  if (!location.href.endsWith('/')) {
    window.location = location.href + '/'
  }
}())

 function download_file(filelink) {
	 
	 	var parts = filelink.split('/');
		var filename = parts.pop() || parts.pop();  // handle potential trailing slash
	 
     var req = new XMLHttpRequest();
     req.open("GET", filelink, true);
     req.responseType = "blob";
     req.onload = function (event) {
         var blob = req.response;
         var fileName = req.getResponseHeader("fileName") //if you have the fileName header available
         var link=document.createElement('a');
         link.href=window.URL.createObjectURL(blob);
         link.download=filename;
         link.click();
     };

     req.send();
 }