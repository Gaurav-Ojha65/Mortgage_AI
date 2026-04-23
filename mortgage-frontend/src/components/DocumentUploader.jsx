import React, { useState, useCallback, useRef } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// =============================================================================
// Styles - Matching existing dark theme
// =============================================================================

const S = {
  container: {
    width: "100%",
    maxWidth: 700,
    margin: "0 auto",
    padding: "1.5rem",
  },
  dropzone: {
    border: "2px dashed #334155",
    borderRadius: 12,
    padding: "2.5rem 1.5rem",
    textAlign: "center",
    background: "#1e293b",
    transition: "all 0.2s ease",
    cursor: "pointer",
  },
  dropzoneActive: {
    borderColor: "#60a5fa",
    background: "rgba(96, 165, 250, 0.1)",
  },
  dropzoneError: {
    borderColor: "#f87171",
    background: "rgba(248, 113, 113, 0.1)",
  },
  icon: {
    fontSize: "2.5rem",
    color: "#64748b",
    marginBottom: "0.75rem",
  },
  title: {
    fontSize: "1.1rem",
    fontWeight: 600,
    color: "#e2e8f0",
    marginBottom: "0.25rem",
  },
  subtitle: {
    fontSize: "0.8rem",
    color: "#64748b",
    marginBottom: "1rem",
  },
  fileTypes: {
    fontSize: "0.75rem",
    color: "#475569",
    marginTop: "0.5rem",
  },
  progressBar: {
    width: "100%",
    height: 6,
    background: "#334155",
    borderRadius: 3,
    overflow: "hidden",
    marginTop: "1rem",
  },
  progressFill: {
    height: "100%",
    background: "linear-gradient(90deg, #60a5fa, #22d3a5)",
    transition: "width 0.3s ease",
  },
  previewCard: {
    background: "#1e293b",
    border: "1px solid #334155",
    borderRadius: 12,
    padding: "1.25rem",
    marginTop: "1.5rem",
  },
  previewTitle: {
    fontSize: "0.95rem",
    fontWeight: 600,
    color: "#f1f5f9",
    marginBottom: "1rem",
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  fieldGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
    gap: "0.75rem",
  },
  fieldRow: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    padding: "0.75rem",
    background: "#0f172a",
    borderRadius: 8,
    border: "1px solid #334155",
  },
  fieldLabel: {
    fontSize: "0.7rem",
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  fieldValue: {
    fontSize: "0.9rem",
    color: "#e2e8f0",
    fontFamily: "monospace",
  },
  fieldInput: {
    background: "#0f172a",
    border: "1px solid #334155",
    borderRadius: 6,
    padding: "0.5rem 0.75rem",
    color: "#e2e8f0",
    fontFamily: "monospace",
    fontSize: "0.85rem",
    outline: "none",
  },
  fieldInputFocus: {
    borderColor: "#60a5fa",
  },
  confidenceBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "0.25rem 0.75rem",
    borderRadius: 9999,
    fontSize: "0.75rem",
    fontWeight: 700,
    fontFamily: "monospace",
  },
  confidenceHigh: {
    background: "rgba(34, 211, 165, 0.15)",
    color: "#22d3a5",
    border: "1px solid rgba(34, 211, 165, 0.4)",
  },
  confidenceMedium: {
    background: "rgba(251, 191, 36, 0.15)",
    color: "#fbbf24",
    border: "1px solid rgba(251, 191, 36, 0.4)",
  },
  confidenceLow: {
    background: "rgba(248, 113, 113, 0.15)",
    color: "#f87171",
    border: "1px solid rgba(248, 113, 113, 0.4)",
  },
  warningBox: {
    background: "rgba(251, 191, 36, 0.1)",
    border: "1px solid rgba(251, 191, 36, 0.3)",
    borderRadius: 8,
    padding: "0.75rem 1rem",
    marginTop: "1rem",
  },
  warningText: {
    fontSize: "0.8rem",
    color: "#fbbf24",
    lineHeight: 1.6,
  },
  buttonGroup: {
    display: "flex",
    gap: "0.75rem",
    marginTop: "1.25rem",
  },
  buttonPrimary: {
    flex: 1,
    background: "linear-gradient(135deg, #60a5fa, #3b82f6)",
    border: "none",
    borderRadius: 8,
    padding: "0.75rem 1.5rem",
    color: "#fff",
    fontFamily: "monospace",
    fontSize: "0.85rem",
    fontWeight: 700,
    cursor: "pointer",
    transition: "all 0.2s",
  },
  buttonSecondary: {
    background: "transparent",
    border: "1px solid #334155",
    borderRadius: 8,
    padding: "0.75rem 1.5rem",
    color: "#64748b",
    fontFamily: "monospace",
    fontSize: "0.8rem",
    cursor: "pointer",
    transition: "all 0.2s",
  },
  spinner: {
    width: 24,
    height: 24,
    border: "3px solid #334155",
    borderTopColor: "#60a5fa",
    borderRadius: "50%",
    animation: "spin 0.7s linear infinite",
  },
  errorBox: {
    background: "rgba(248, 113, 113, 0.1)",
    border: "1px solid rgba(248, 113, 113, 0.3)",
    borderRadius: 8,
    padding: "1rem",
    marginTop: "1rem",
  },
  errorText: {
    fontSize: "0.85rem",
    color: "#f87171",
  },
};

// =============================================================================
// Helper Functions
// =============================================================================

const formatCurrency = (value) => {
  if (value === null || value === undefined) return "N/A";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

const getConfidenceStyle = (confidence) => {
  if (confidence >= 0.8) return { badge: S.confidenceHigh, label: "High" };
  if (confidence >= 0.6) return { badge: S.confidenceMedium, label: "Medium" };
  return { badge: S.confidenceLow, label: "Low" };
};

const FIELD_LABELS = {
  annual_income: "Annual Income",
  monthly_income: "Monthly Income",
  employer_name: "Employer Name",
  pay_period: "Pay Period",
  bank_balance: "Bank Balance",
  gross_pay_ytd: "Gross Pay (YTD)",
  net_pay: "Net Pay",
  employee_name: "Employee Name",
};

// =============================================================================
// Main Component
// =============================================================================

export default function DocumentUploader({ onFieldsExtracted, autoApply = false }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [extractionResult, setExtractionResult] = useState(null);
  const [error, setError] = useState(null);
  const [editableFields, setEditableFields] = useState({});
  const fileInputRef = useRef(null);

  // Handle drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Handle file drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setError(null);

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, []);

  // Handle file selection via input
  const handleChange = useCallback((e) => {
    e.preventDefault();
    const files = e.target?.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  }, []);

  // Validate and upload file
  const handleFile = async (file) => {
    setError(null);

    // Validate file type
    const validTypes = [
      "application/pdf",
      "image/png",
      "image/jpeg",
      "image/jpg",
    ];
    if (!validTypes.includes(file.type)) {
      setError("Invalid file type. Please upload PDF, PNG, or JPG.");
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
      setError("File too large. Maximum size is 10MB.");
      return;
    }

    // Upload and extract
    await uploadAndExtract(file);
  };

  // Upload file to API and extract
  const uploadAndExtract = async (file) => {
    setUploading(true);
    setUploadProgress(0);
    setExtractionResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Simulate progress (fetch doesn't provide upload progress)
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => Math.min(prev + 10, 90));
      }, 200);

      const response = await fetch(`${API_URL}/api/documents/upload`, {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed (${response.status})`);
      }

      const result = await response.json();

      setExtractionResult(result);
      setEditableFields(result.extracted_fields || {});

      // Auto-apply if enabled and confidence is high
      if (autoApply && result.confidence >= 0.8 && onFieldsExtracted) {
        setTimeout(() => {
          handleApplyFields();
        }, 1000);
      }
    } catch (err) {
      setError(err.message || "Failed to process document");
      setExtractionResult(null);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 500);
    }
  };

  // Handle field edit
  const handleFieldChange = (field, value) => {
    setEditableFields((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  // Apply extracted fields to parent form
  const handleApplyFields = () => {
    if (onFieldsExtracted) {
      onFieldsExtracted(editableFields);
    }
  };

  // Reset and try again
  const handleReset = () => {
    setExtractionResult(null);
    setError(null);
    setEditableFields({});
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Render dropzone content
  const renderDropzoneContent = () => {
    if (uploading) {
      return (
        <>
          <div style={S.spinner} />
          <p style={{ color: "#60a5fa", fontSize: "0.9rem", marginTop: "1rem", fontFamily: "monospace" }}>
            Processing document...
          </p>
          <div style={S.progressBar}>
            <div style={{ ...S.progressFill, width: `${uploadProgress}%` }} />
          </div>
        </>
      );
    }

    return (
      <>
        <div style={S.icon}>📄</div>
        <p style={S.title}>
          Drop your document here
        </p>
        <p style={S.subtitle}>
          Pay stub, bank statement, or tax return
        </p>
        <p style={S.fileTypes}>
          Supported: PDF, PNG, JPG (max 10MB)
        </p>
      </>
    );
  };

  // Render confidence badge
  const renderConfidenceBadge = (confidence) => {
    const { badge, label } = getConfidenceStyle(confidence);
    return (
      <span style={{ ...S.confidenceBadge, ...badge }}>
        <span style={{ width: 6, height: 6, borderRadius: "50%", background: "currentColor" }} />
        {label} ({(confidence * 100).toFixed(0)}%)
      </span>
    );
  };

  // Render field row
  const renderFieldRow = (field, value) => {
    if (value === null || value === undefined) return null;

    const isCurrency = field.includes("income") || field.includes("balance") || field.includes("pay");
    const displayValue = isCurrency ? formatCurrency(value) : value;

    return (
      <div key={field} style={S.fieldRow}>
        <label style={S.fieldLabel}>{FIELD_LABELS[field] || field}</label>
        <input
          type="text"
          value={editableFields[field] ?? ""}
          onChange={(e) => handleFieldChange(field, e.target.value)}
          style={S.fieldInput}
          placeholder="Not extracted"
        />
      </div>
    );
  };

  return (
    <div style={S.container}>
      {/* Dropzone */}
      <div
        style={{
          ...S.dropzone,
          ...(dragActive ? S.dropzoneActive : {}),
          ...(error ? S.dropzoneError : {}),
        }}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg"
          onChange={handleChange}
          style={{ display: "none" }}
          disabled={uploading}
        />
        {renderDropzoneContent()}
      </div>

      {/* Error message */}
      {error && (
        <div style={S.errorBox}>
          <p style={S.errorText}>⚠ {error}</p>
        </div>
      )}

      {/* Extraction result */}
      {extractionResult && (
        <div style={S.previewCard}>
          <div style={S.previewTitle}>
            <span>📋 Extracted Data</span>
            {renderConfidenceBadge(extractionResult.confidence)}
          </div>

          {/* Document info */}
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", fontSize: "0.75rem", color: "#64748b" }}>
            <span>Type: {extractionResult.document_type}</span>
            <span>•</span>
            <span>Pages: {extractionResult.pages_processed}</span>
            <span>•</span>
            <span>Method: {extractionResult.extraction_method}</span>
          </div>

          {/* Field grid */}
          <div style={S.fieldGrid}>
            {Object.entries(extractionResult.extracted_fields || {}).map(([field, value]) =>
              renderFieldRow(field, value)
            )}
          </div>

          {/* Warnings */}
          {extractionResult.warnings?.length > 0 && (
            <div style={S.warningBox}>
              <p style={{ ...S.warningText, fontWeight: 600, marginBottom: "0.5rem" }}>
                ⚠ Extraction Warnings
              </p>
              <ul style={{ ...S.warningText, margin: 0, paddingLeft: "1.25rem" }}>
                {extractionResult.warnings.map((warning, i) => (
                  <li key={i}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Raw text preview */}
          {extractionResult.raw_text_preview && (
            <details style={{ marginTop: "1rem" }}>
              <summary style={{ fontSize: "0.75rem", color: "#64748b", cursor: "pointer" }}>
                View raw text preview
              </summary>
              <pre style={{
                background: "#0f172a",
                borderRadius: 6,
                padding: "0.75rem",
                fontSize: "0.7rem",
                color: "#94a3b8",
                marginTop: "0.5rem",
                maxHeight: 200,
                overflow: "auto",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}>
                {extractionResult.raw_text_preview}
              </pre>
            </details>
          )}

          {/* Action buttons */}
          <div style={S.buttonGroup}>
            <button
              onClick={handleApplyFields}
              style={S.buttonPrimary}
              disabled={uploading}
            >
              ✓ Use these values
            </button>
            <button
              onClick={handleReset}
              style={S.buttonSecondary}
              disabled={uploading}
            >
              Upload different document
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
