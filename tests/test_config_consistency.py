"""Tests for configuration consistency checks."""

import importlib.util
import os
import pytest

# Import check.py directly without loading the full Tools package (which has
# heavy dependencies like matplotlib) so that this test file can run in
# lightweight environments.
_check_spec = importlib.util.spec_from_file_location(
    "check",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Tools", "check.py"),
)
check = importlib.util.module_from_spec(_check_spec)
_check_spec.loader.exec_module(check)


class MockConfig:
    """A minimal mock config object for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestCheckConfigConsistency:
    """Tests for check.check_config_consistency."""

    def test_valid_config_no_format(self):
        """No error when resp has no format key."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=False, old_cluster=True)
        varlist = {"resp": {}}
        check.check_config_consistency(config, varlist)  # should not raise

    def test_valid_config_compressed_format(self):
        """No error when format is 'compressed' and leave_one_out_cv is False."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=False, old_cluster=True)
        varlist = {"resp": {"format": "compressed"}}
        check.check_config_consistency(config, varlist)  # should not raise

    def test_unstructured_format_requires_loocv(self):
        """Error when format is 'unstructured' and leave_one_out_cv is False."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=False, old_cluster=True)
        varlist = {"resp": {"format": "unstructured"}}
        with pytest.raises(ValueError, match="unstructured"):
            check.check_config_consistency(config, varlist)

    def test_unstructured_format_with_loocv_valid(self):
        """No error when format is 'unstructured' and leave_one_out_cv is True."""
        config = MockConfig(leave_one_out_cv=True, sel_most_PFT_sites=False, old_cluster=True)
        varlist = {"resp": {"format": "unstructured"}}
        check.check_config_consistency(config, varlist)  # should not raise

    def test_sel_most_pft_sites_requires_old_cluster_false(self):
        """Error when sel_most_PFT_sites=True and old_cluster=True."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=True, old_cluster=True)
        varlist = {"resp": {}}
        with pytest.raises(ValueError, match="sel_most_PFT_sites"):
            check.check_config_consistency(config, varlist)

    def test_sel_most_pft_sites_with_old_cluster_false_valid(self):
        """No error when sel_most_PFT_sites=True and old_cluster=False."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=True, old_cluster=False)
        varlist = {"resp": {}}
        check.check_config_consistency(config, varlist)  # should not raise

    def test_multiple_errors_reported_together(self):
        """Both errors are reported in a single ValueError."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=True, old_cluster=True)
        varlist = {"resp": {"format": "unstructured"}}
        with pytest.raises(ValueError) as exc_info:
            check.check_config_consistency(config, varlist)
        message = str(exc_info.value)
        assert "unstructured" in message
        assert "sel_most_PFT_sites" in message

    def test_missing_config_attributes_use_defaults(self):
        """Missing optional config attributes do not cause errors (use defaults)."""
        config = MockConfig(leave_one_out_cv=False)
        varlist = {"resp": {}}
        check.check_config_consistency(config, varlist)  # should not raise

    def test_varlist_missing_resp_key(self):
        """No error when varlist has no 'resp' key."""
        config = MockConfig(leave_one_out_cv=False, sel_most_PFT_sites=False, old_cluster=True)
        varlist = {}
        check.check_config_consistency(config, varlist)  # should not raise
