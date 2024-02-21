function add_trailing_slash() {
	
    //If there is no trailing shash after the path in the url add it
    if (window.location.pathname.endsWith('/') === false) {
        var url = window.location.protocol + '//' + 
                window.location.host + 
                window.location.pathname + '/' + 
                window.location.search;

        window.history.replaceState(null, document.title, url);
    }
}

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