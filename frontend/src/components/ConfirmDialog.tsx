import React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { X } from 'lucide-react'

interface ConfirmDialogProps {
  open: boolean
  title?: string
  description?: string
  confirmLabel?: string
  cancelLabel?: string
  confirmVariant?: 'danger' | 'primary'
  onConfirm: () => void
  onCancel: () => void
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  title = 'Are you sure?',
  description,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  confirmVariant = 'danger',
  onConfirm,
  onCancel,
}) => {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-[var(--ui-overlay)]"
          role="dialog"
          aria-modal="true"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="w-full sm:max-w-md m-0 sm:m-4 card-glass rounded-2xl sm:rounded-3xl overflow-hidden"
            initial={{ y: 40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 40, opacity: 0 }}
            transition={{ duration: 0.2 }}
            role="document"
          >
            <div className="p-4 sm:p-6 border-b border-ui-border-muted flex items-center justify-between">
              <div className="text-sm font-semibold text-ui-text-primary">{title}</div>
              <button
                className="p-2 rounded-xl bg-ui-surface-elevated/60 hover:bg-ui-surface-elevated/80 transition-colors"
                aria-label="Close"
                onClick={onCancel}
              >
                <X className="w-4 h-4 text-ui-text-secondary" />
              </button>
            </div>
            {description && (
              <div className="p-4 sm:p-6 text-sm text-ui-text-secondary">{description}</div>
            )}
            <div className="p-4 sm:p-6 pt-0 sm:pt-0 flex gap-2 justify-end">
              <button
                onClick={onCancel}
                className="h-10 px-4 rounded-xl btn-outline-neutral font-medium"
              >
                {cancelLabel}
              </button>
              <button
                onClick={onConfirm}
                className={`h-10 px-4 rounded-xl font-medium ${
                  confirmVariant === 'danger'
                    ? 'btn-solid-danger'
                    : 'bg-ui-brand-primary text-white border border-token-brand-40 rounded-xl'
                }`}
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default ConfirmDialog


