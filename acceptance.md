# Acceptance Criteria

## Task 1: Logging Configuration

### Acceptance Criteria
- [ ] setup_logger(name=None) returns root logger when name is None
- [ ] setup_logger(name="module_name") returns named logger
- [ ] Logger format includes timestamp, level, filename, line number, and message
- [ ] Rank 0 logs at INFO level by default
- [ ] Non-zero ranks log at WARNING level by default
- [ ] basicConfig is only called once (no duplicate handlers)
- [ ] Datetime format is "YYYY-MM-DD HH:MM:SS"
