function handler(event) {
  var request = event.request;
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
