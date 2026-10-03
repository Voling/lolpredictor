var OPEN = { "/api/recent": true };
var OPEN_PREFIXES = ["/api/shared/"];

function open(uri) {
  if (OPEN[uri]) return true;
  for (var i = 0; i < OPEN_PREFIXES.length; i++) {
    if (uri.indexOf(OPEN_PREFIXES[i]) === 0 && uri.length > OPEN_PREFIXES[i].length) return true;
  }
  return false;
}

function handler(event) {
  var request = event.request;
  if (open(request.uri) && (request.method === "GET" || request.method === "HEAD")) {
    return request;
  }
  var token = request.headers["x-auth"];
  if (token && /^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$/.test(token.value) && token.value.length < 8192) {
    return request;
  }
  return {
    statusCode: 401,
    statusDescription: "Unauthorized",
    headers: { "cache-control": { value: "no-store" } },
  };
}
