(function() {
  if (!location.href.endsWith('/')) {
    window.location = location.href + '/'
  }
}())