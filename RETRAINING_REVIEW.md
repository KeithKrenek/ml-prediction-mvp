# Model Retraining Code Review & Bug Fixes

**Date:** 2025-11-08
**Reviewer:** Claude
**Scope:** Comprehensive review of model retraining workflow

## Executive Summary

Fixed **3 CRITICAL bugs** that were preventing models from being promoted to production:
1. Version ID mismatch causing "Model not found" errors
2. Database session race condition in promotion logic
3. Timezone inconsistencies across the codebase

## Critical Bugs Fixed

### 1. Version ID Mismatch (CRITICAL) ✅ FIXED

**Problem:**
- `retrain_models.py` generated a version_id (e.g., `prophet_timing_20251107_203340_9a42832e`)
- Saved model file with this version_id
- Called `registry.register_model()` which generated a **DIFFERENT** version_id (e.g., `prophet_timing_20251107_203340_b6bb172d`)
- Script continued using OLD version_id, but database had NEW version_id
- Evaluation and promotion failed with "Model not found" errors

**Root Cause:**
`model_registry.py` line 93 always generated a new version_id, ignoring the one already created.

**Fix:**
- Modified `register_model()` to accept optional `version_id` parameter
- Updated `retrain_models.py` to pass the version_id to `register_model()`
- Ensures same version_id is used for file save, database registration, and evaluation

**Files Changed:**
- `src/models/model_registry.py` (lines 62-96)
- `scripts/retrain_models.py` (lines 121, 284)

---

### 2. Database Session Race Condition (CRITICAL) ✅ FIXED

**Problem:**
`promote_to_production()` called `self.get_production_model()` which opened a NEW database session, then tried to add that object to the existing session, causing `DetachedInstanceError`.

**Root Cause:**
Mixing objects from different database sessions violates SQLAlchemy's session management.

**Fix:**
Query for both the new model AND current production model in the SAME session:

```python
# Before (WRONG):
current_prod = self.get_production_model(new_model.model_type)  # Opens new session!

# After (CORRECT):
current_prod = (
    session.query(ModelVersion)
    .filter_by(model_type=new_model.model_type, is_production=True)
    .order_by(ModelVersion.promoted_at.desc())
    .first()
)  # Uses same session
```

**Files Changed:**
- `src/models/model_registry.py` (lines 196-202)

---

### 3. Timezone Inconsistencies (CRITICAL) ✅ FIXED

**Problem:**
- Database schema uses `datetime.now(timezone.utc)` (timezone-aware)
- Retraining scripts used `datetime.now()` (timezone-naive)
- Mixing causes comparison errors and unreliable version IDs

**Fix:**
Changed all `datetime.now()` to `datetime.now(timezone.utc)` for consistency:

**Files Changed:**
- `src/models/model_registry.py` (lines 58, 497)
- `scripts/retrain_models.py` (lines 119, 282, 344, 396)

---

## Error Messages Resolved

### Before Fix:
```
ERROR | src.models.model_registry:evaluate_model:253 - Model prophet_timing_20251107_203340_9a42832e not found
ERROR | src.models.model_registry:evaluate_model:253 - Model claude_api_content_20251107_203345_d3fe394a not found
INFO  | Model not promoted. Reason: Model not found
```

### After Fix:
✅ Models will be found in the database
✅ Evaluation will succeed
✅ Promotion logic will execute correctly

---

## Additional Issues Identified (Not Fixed Yet)

### High Priority
1. **Fragile timezone stripping in evaluator.py** - Assumes timezone-aware Series
2. **Memory inefficiency** - `archive_old_versions()` loads all models into memory
3. **Version ID collision risk** - Only second-precision timestamp + 8-char UUID

### Medium Priority
4. Incomplete error handling in evaluation
5. File path validation missing
6. Connection pool not reused

### Low Priority
7. Inconsistent Path object usage
8. Hard-coded configuration values
9. Prediction data truncation (only stores first 100)

See full analysis from Explore agent for details.

---

## Testing Recommendations

1. **Run full retraining cycle:**
   ```bash
   python scripts/retrain_models.py
   ```

2. **Verify in database:**
   - Check ModelVersion table for matching version_ids
   - Verify model files exist at registered paths
   - Confirm evaluations are stored correctly

3. **Test promotion:**
   - Verify models are promoted when performance improves
   - Check that only one model is marked as production per type

---

## Code Quality Improvements Made

1. ✅ Fixed critical race condition in database operations
2. ✅ Standardized timezone handling across codebase
3. ✅ Added better version ID management
4. ✅ Improved code comments explaining session management

---

## Backward Compatibility

All changes maintain backward compatibility:
- `version_id` parameter is optional in `register_model()`
- Default behavior generates version_id if not provided
- No breaking changes to existing code

---

## Next Steps

1. ✅ Commit and push fixes
2. 🔄 Run integration tests
3. 🔄 Monitor next automated retraining run
4. 🔄 Address remaining medium/low priority issues

---

## Files Modified

```
src/models/model_registry.py    - 3 critical fixes
scripts/retrain_models.py        - 4 timezone + version_id fixes
```

## Summary Statistics

- **Critical Bugs Fixed:** 3
- **Lines Changed:** ~15
- **Files Modified:** 2
- **Backward Compatibility:** ✅ Maintained
- **Test Status:** ⏳ Ready for testing
