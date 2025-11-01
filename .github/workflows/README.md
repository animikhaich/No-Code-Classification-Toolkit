# GitHub Workflows

This directory contains GitHub Actions workflows for the No-Code Classification Toolkit.

## Docker Build and Push Workflow

**File:** `docker-build-push.yml`

### Purpose
Automatically builds and pushes Docker images to GitHub Container Registry (ghcr.io) whenever code is pushed to the repository.

### Triggers
The workflow runs on:
- **Push to main branch**: Builds and pushes images with `latest` tag
- **Push tags matching `v*`**: Builds and pushes versioned releases (e.g., `v1.0.0`)
- **Pull requests to main**: Builds images for testing (does not push)
- **Manual trigger**: Can be triggered manually via GitHub Actions UI

### Docker Image Tags
The workflow creates multiple tags for each build:

- `latest` - Latest build from the main branch
- `main` - Latest build from the main branch
- `v1.2.3` - Semantic version tags (for tagged releases)
- `v1.2` - Major.minor version tags
- `v1` - Major version tags
- `main-<sha>` - Branch name with commit SHA

### Docker Registry
Images are pushed to GitHub Container Registry:
```
ghcr.io/animikhaich/no-code-classification-toolkit
```

### Usage

#### Pull the latest image:
```bash
docker pull ghcr.io/animikhaich/no-code-classification-toolkit:latest
```

#### Pull a specific version:
```bash
docker pull ghcr.io/animikhaich/no-code-classification-toolkit:v1.0.0
```

#### Run the container:
```bash
docker run -it --runtime nvidia --net host -v /path/to/dataset:/data ghcr.io/animikhaich/no-code-classification-toolkit:latest
```

### Permissions
The workflow requires:
- `contents: read` - To checkout the repository
- `packages: write` - To push images to GitHub Container Registry

### Features
- **Docker Buildx**: Uses BuildKit for efficient multi-platform builds
- **Layer caching**: Leverages GitHub Actions cache for faster builds
- **Metadata extraction**: Automatically generates Docker labels and tags
- **Security**: Uses `GITHUB_TOKEN` for authentication (no manual secrets needed)

### Monitoring
You can monitor workflow runs in the [Actions tab](https://github.com/animikhaich/No-Code-Classification-Toolkit/actions) of the repository.
