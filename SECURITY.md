# Security Summary

## Overview
This document addresses security considerations for the No-Code Classification Toolkit.

## Security Analysis Results

### CodeQL Findings

**Path Injection Alerts (3 instances)**
- **Location**: `core/data_loader_pytorch.py` and `core/data_loader.py`
- **Status**: Acknowledged - By Design
- **Details**: The application requires users to provide dataset directory paths as part of its core functionality

#### Context
This toolkit is designed to be run in a containerized environment where:
1. Users mount their own dataset directories
2. The application runs in an isolated Docker container
3. Users have full control over the container and its file system access

#### Mitigations Implemented
1. **Path Normalization**: All user-provided paths are normalized using `os.path.normpath()` to remove redundant separators and resolve relative path components
2. **Label Validation**: Class directory names are validated to prevent path traversal attempts:
   ```python
   if '..' in label or '/' in label or '\\' in label:
       raise ValueError(f"Invalid class directory name: {label}")
   ```
3. **Directory Verification**: Paths are validated to ensure they point to actual directories before processing
4. **Container Isolation**: The application runs in a Docker container with user-controlled volume mounts

#### Risk Assessment
- **Risk Level**: Low
- **Rationale**: 
  - The application is designed for single-user, local execution
  - Users are providing paths to their own data
  - Container isolation prevents access to host system files outside mounted volumes
  - No network-accessible API that could be exploited remotely

## Best Practices Implemented

### General Security
- ✅ Input validation on all user-provided parameters
- ✅ Error handling to prevent information leakage
- ✅ No hardcoded credentials
- ✅ Dependencies pinned to specific versions
- ✅ Container isolation for runtime environment

### Data Security
- ✅ Read-only access to dataset directories (user controls write permissions via mount)
- ✅ No sensitive data stored in logs
- ✅ Model weights and logs saved to user-specified locations

### Code Security
- ✅ No use of `eval()` or `exec()` on user input (except for controlled model initialization)
- ✅ Secure random number generation for data augmentation
- ✅ Type hints and validation throughout codebase

## Deployment Recommendations

### For Production Use
1. **Container Security**:
   - Use read-only filesystem for container (`--read-only` flag)
   - Mount only necessary directories
   - Run container with limited user permissions (non-root)
   - Use security scanning tools on Docker images

2. **Network Security**:
   - Run on isolated networks
   - Use `--net host` only when necessary
   - Consider using reverse proxy for web interface if exposed

3. **Data Security**:
   - Ensure dataset directories have appropriate permissions
   - Use encrypted volumes for sensitive data
   - Regularly backup trained models

4. **Resource Limits**:
   - Set memory limits (`--memory` flag)
   - Set CPU limits (`--cpus` flag)
   - Monitor resource usage

### Example Secure Docker Run Command
```bash
docker run -it \
  --gpus all \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /app/model \
  --tmpfs /app/logs \
  -v /path/to/dataset:/data:ro \
  -v /path/to/output:/output \
  --memory=16g \
  --cpus=4 \
  --user $(id -u):$(id -g) \
  animikhaich/zero-code-classifier:pytorch
```

## Vulnerability Management

### Dependency Updates
- Regularly update dependencies to latest stable versions
- Monitor security advisories for PyTorch, TensorFlow, and other dependencies
- Use automated tools like Dependabot for dependency updates

### Known Limitations
1. **Pickle Files**: PyTorch uses pickle for model serialization which can be unsafe with untrusted data
   - **Mitigation**: Only load models you have trained yourself
2. **User Input**: Application accepts arbitrary file paths
   - **Mitigation**: Run in containerized environment with limited filesystem access

## Security Checklist for Users

- [ ] Run container with minimal required permissions
- [ ] Use read-only mounts for dataset directories
- [ ] Regularly update Docker images
- [ ] Monitor container resource usage
- [ ] Backup trained models securely
- [ ] Review container logs for anomalies
- [ ] Use separate containers for different projects
- [ ] Clean up temporary files after training

## Reporting Security Issues

If you discover a security vulnerability, please email: animikhaich@gmail.com

**Do not** create public issues for security vulnerabilities.

## Compliance

This application:
- ✅ Does not collect or transmit user data
- ✅ Runs entirely locally or in user-controlled environments
- ✅ Does not require network access for core functionality
- ✅ Stores all data in user-specified locations
- ✅ Provides transparency through open source code

## Conclusion

The identified path injection alerts are inherent to the application's design and purpose. The implemented mitigations, combined with containerization and proper deployment practices, provide adequate security for the intended use case of local, single-user image classification model training.
