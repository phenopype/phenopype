Gallery templates
=================


.. raw:: html

	<script> 

			function download(url) {
				const a = document.createElement('a')
				a.href = url
				a.download = url.split('/').pop()
				document.body.appendChild(a)
				a.click()
				document.body.removeChild(a)
			}
			
		invoke = (event) => {
			let nameOfFunction = this[event.target.name];
			let arg1 = event.target.getAttribute('github_url');
			// We can add more arguments as needed...
			window[nameOfFunction](arg1)
			// Hope the function is in the window.
			// Else the respective object need to be used
			})
		}
			
	</script>
	
	
		<button type="button"
        className="btn btn-default"
        onClick="invoke"
        name='download'
        github_url='https://raw.githubusercontent.com/phenopype/phenopype-templates/main/templates/detection/single1.yaml'>Download</button>



	<button onclick='download(https://raw.githubusercontent.com/phenopype/phenopype-templates/main/templates/detection/single1.yaml)'>
			Download yaml
	</button>
