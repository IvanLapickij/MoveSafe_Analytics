# Use the prebuilt NGINX-RTMP server
FROM alfg/nginx-rtmp:latest

# Expose RTMP port (1935) and HTTP Stats port (8080)
EXPOSE 1935
EXPOSE 8080

# Copy the custom RTMP configuration file
COPY nginx.conf /etc/nginx/nginx.conf
