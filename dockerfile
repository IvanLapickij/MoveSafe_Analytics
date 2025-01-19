FROM alfg/nginx-rtmp:latest

# Expose RTMP and HTTP ports
EXPOSE 1935
EXPOSE 8080

# Copy custom nginx configuration into the container
COPY nginx.conf /etc/nginx/nginx.conf
