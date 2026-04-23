# Frontend Dockerfile - Production Ready
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY mortgage-frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source and build
COPY mortgage-frontend/ .
RUN npm run build

# Production stage with nginx
FROM nginx:1.25-alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user
RUN chown -R nginx:nginx /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
