var ACCOUNT = { "/api/me": true, "/api/history": true, "/api/link": true, "/api/link/verify": true, "/api/status": true };
var ACCOUNT_PREFIXES = ["/api/history/"];

function needsAccount(uri) {
  if (ACCOUNT[uri]) return true;
  for (var i = 0; i < ACCOUNT_PREFIXES.length; i++) {
    if (uri.indexOf(ACCOUNT_PREFIXES[i]) === 0) return true;
  }
  return false;
}

function handler(event) {
  var request = event.request;
  request.headers["x-viewer-ip"] = { value: event.viewer.ip };
  if (!needsAccount(request.uri)) {
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
