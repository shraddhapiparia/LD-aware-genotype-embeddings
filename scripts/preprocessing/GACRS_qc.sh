#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Step-by-step PLINK2 pipeline + per-step variant counts
# Produces 3 bfile sets:
#  1) QC'ed before LD:        02142026/garcs_qc_autosomes_qced.*
#  2) KING-filtered samples:  02142026/garcs_qc_autosomes_qced.king_unrel.*
#     (samples filtered using KING in.id, variants = full QC'ed set)
#  3) LD-pruned + KING:       02142026/garcs_qc_autosomes_qced.king_unrel.ldpruned.*
#     (samples filtered + LD-pruned variants)
#
# Also writes:
#  - a global summary log with counts: 02142026/global_unrel_kids_pipeline.log
#  - each plink2 step's native .log (as usual)
###############################################################################

# ---- user inputs ----
PLINK2="${PLINK2:-./plink2}"

BFILE="/proj/regeps/regep00/studies/CRA/data/imputed/2013_CRA_GWAS_UMich/data/freezes/20211210_GRCh38/2013_CRA_GWAS_FULL_MERGE"
OUTDIR="02142026"

# Step 2 keep file (phenotyped subjects)
PHENO_KEEP="GACRS_filtered_QC/keep_pheno_subjects.txt"

# QC thresholds (Step 1)
MAF=0.05
GENO=0.05

# LD pruning (Step 3)
LD_WINDOW_KB=50
LD_STEP=5
LD_R2=0.2

# KING cutoff (Step 4)
KING_CUTOFF=0.0442
# ----------------------

mkdir -p "${OUTDIR}"

LOGFILE="${OUTDIR}/global_unrel_kids_pipeline.log"
: > "${LOGFILE}"

ts() { date "+%Y-%m-%d %H:%M:%S %Z"; }

count_bim() { [[ -f "$1" ]] && wc -l < "$1" || echo 0; }
count_fam() { [[ -f "$1" ]] && wc -l < "$1" || echo 0; }
count_lines() { [[ -f "$1" ]] && wc -l < "$1" || echo 0; }

log() {
  echo "[$(ts)] $*" | tee -a "${LOGFILE}" >/dev/null
}

# Grep helpers (best-effort parsing from plink2 logs)
grep_first_num() {
  # usage: grep_first_num "pattern" logfile
  local pat="$1" file="$2"
  grep -E "$pat" "$file" | head -n 1 | grep -Eo '[0-9]+' | head -n 1 || true
}
grep_all_nums_last() {
  # usage: grep_all_nums_last "pattern" logfile
  local pat="$1" file="$2"
  grep -E "$pat" "$file" | tail -n 1 | grep -Eo '[0-9]+' | tail -n 1 || true
}

###############################################################################
# Prefixes 
###############################################################################
STEP1_OUT="${OUTDIR}/garcs_qc_autosomes_manual_test"
STEP2_OUT="${OUTDIR}/garcs_qc_autosomes_qced"
LD_OUT="${OUTDIR}/ldprune_autosomes_w${LD_WINDOW_KB}_${LD_STEP}_p${LD_R2}"
KING_OUT="${OUTDIR}/king_autosomes_ldpruned_cutoff_p0442"  

# Derived files
PRUNE_IN="${LD_OUT}.prune.in"
PRUNE_OUT="${LD_OUT}.prune.out"

KING_IN_ID="${KING_OUT}.king.cutoff.in.id"
KING_OUT_ID="${KING_OUT}.king.cutoff.out.id"

# Final bfiles
KING_FILTERED_OUT="${OUTDIR}/garcs_qc_autosomes_qced_king_unrel"
KING_LDPRUNED_OUT="${OUTDIR}/garcs_qc_autosomes_qced_king_unrel_ldpruned"

###############################################################################
# Header
###############################################################################
log "=== START PIPELINE ==="
log "PLINK2: ${PLINK2}"
log "Raw input BFILE: ${BFILE}"
log "Phenotype keep file: ${PHENO_KEEP}"
log "OUTDIR: ${OUTDIR}"
log "QC: MAF>=${MAF}, GENO<=${GENO}"
log "LD: window_kb=${LD_WINDOW_KB}, step=${LD_STEP}, r2=${LD_R2}"
log "KING cutoff: ${KING_CUTOFF}"
echo "" >> "${LOGFILE}"

###############################################
# STEP 1: Autosomes QC + set missing var IDs
###############################################
log "---- STEP 1: QC autosomes (geno/maf) + set-missing-var-ids ----"
log "Command:"
log "  ${PLINK2} --allow-extra-chr --bfile ${BFILE} --chr 1-22 --geno ${GENO} --maf ${MAF} --make-bed --out ${STEP1_OUT} --set-missing-var-ids @:#:\$1\$2"

"${PLINK2}" \
  --allow-extra-chr \
  --bfile "${BFILE}" \
  --chr 1-22 \
  --geno "${GENO}" \
  --maf "${MAF}" \
  --make-bed \
  --out "${STEP1_OUT}" \
  --set-missing-var-ids @:#:'$1$2'

STEP1_LOG="${STEP1_OUT}.log"

# counts from log + files
STEP1_VARIANTS_OUT=$(count_bim "${STEP1_OUT}.bim")
STEP1_SAMPLES_OUT=$(count_fam "${STEP1_OUT}.fam")
STEP1_GENO_REMOVED=$(grep_all_nums_last "--geno: .* variants removed" "${STEP1_LOG}")
STEP1_MAF_REMOVED=$(grep_all_nums_last "variants removed due to allele frequency threshold" "${STEP1_LOG}")

log "STEP 1 results:"
log "  Samples saved (fam lines):   ${STEP1_SAMPLES_OUT}"
log "  Variants saved (bim lines):  ${STEP1_VARIANTS_OUT}"
log "  Variants removed by --geno:  ${STEP1_GENO_REMOVED:-NA}"
log "  Variants removed by --maf:   ${STEP1_MAF_REMOVED:-NA}"
echo "" >> "${LOGFILE}"

###############################################################################
# STEP 2: Keep phenotyped subjects + SNPs only + max-alleles 2 
###############################################################################
log "---- STEP 2: Keep phenotyped subjects + biallelic SNPs (ACGT only) ----"
log "Command:"
log "  ${PLINK2} --bfile ${STEP1_OUT} --keep ${PHENO_KEEP} --make-bed --max-alleles 2 --out ${STEP2_OUT} --snps-only just-acgt"

"${PLINK2}" \
  --bfile "${STEP1_OUT}" \
  --keep "${PHENO_KEEP}" \
  --make-bed \
  --max-alleles 2 \
  --out "${STEP2_OUT}" \
  --snps-only just-acgt

STEP2_LOG="${STEP2_OUT}.log"

STEP2_VARIANTS_OUT=$(count_bim "${STEP2_OUT}.bim")
STEP2_SAMPLES_OUT=$(count_fam "${STEP2_OUT}.fam")
STEP2_SAMPLES_KEPT=$(grep_all_nums_last "--keep: .* samples remaining" "${STEP2_LOG}")

log "STEP 2 results (this is your #1 requested bfile set):"
log "  Output prefix:              ${STEP2_OUT}"
log "  Samples saved (fam lines):  ${STEP2_SAMPLES_OUT} (log --keep kept: ${STEP2_SAMPLES_KEPT:-NA})"
log "  Variants saved (bim lines): ${STEP2_VARIANTS_OUT}"
echo "" >> "${LOGFILE}"

###############################################################################
# STEP 3: LD pruning with make-founders 
###############################################################################
log "---- STEP 3: LD pruning (make-founders + indep-pairwise) ----"
log "Command:"
log "  ${PLINK2} --bfile ${STEP2_OUT} --make-founders --indep-pairwise ${LD_WINDOW_KB} ${LD_STEP} ${LD_R2} --out ${LD_OUT}"

"${PLINK2}" \
  --bfile "${STEP2_OUT}" \
  --make-founders \
  --indep-pairwise "${LD_WINDOW_KB}" "${LD_STEP}" "${LD_R2}" \
  --out "${LD_OUT}"

LD_LOG="${LD_OUT}.log"

PRUNE_IN_N=$(count_lines "${PRUNE_IN}")
PRUNE_OUT_N=$(count_lines "${PRUNE_OUT}")
LD_REMOVED=$(grep_all_nums_last "--indep-pairwise .* variants removed" "${LD_LOG}")

log "STEP 3 results:"
log "  Prune-in variants kept (lines):   ${PRUNE_IN_N}"
log "  Prune-out variants removed (lines): ${PRUNE_OUT_N}"
log "  Log-reported variants removed:    ${LD_REMOVED:-NA}"
log "  Prune-in file:  ${PRUNE_IN}"
log "  Prune-out file: ${PRUNE_OUT}"
echo "" >> "${LOGFILE}"

###############################################################################
# STEP 4: KING cutoff on LD-pruned variants, output just fam 
###############################################################################
log "---- STEP 4: KING cutoff on LD-pruned variants (make-just-fam) ----"
log "Command:"
log "  ${PLINK2} --bfile ${STEP2_OUT} --extract ${PRUNE_IN} --king-cutoff ${KING_CUTOFF} --make-just-fam --out ${KING_OUT}"

"${PLINK2}" \
  --bfile "${STEP2_OUT}" \
  --extract "${PRUNE_IN}" \
  --king-cutoff "${KING_CUTOFF}" \
  --make-just-fam \
  --out "${KING_OUT}"

KING_LOG="${KING_OUT}.log"

KING_VARIANTS_USED=$(grep_all_nums_last "--extract: .* variants remaining" "${KING_LOG}")
UNREL_N=$(count_lines "${KING_IN_ID}")
REL_EXCLUDED_N=$(count_lines "${KING_OUT_ID}")

log "STEP 4 results:"
log "  Variants used for KING (after extract): ${KING_VARIANTS_USED:-NA}"
log "  Unrelated samples kept (in.id lines):  ${UNREL_N}"
log "  Related/excluded samples (out.id lines): ${REL_EXCLUDED_N}"
log "  KING keep list: ${KING_IN_ID}"
echo "" >> "${LOGFILE}"

###############################################################################
# STEP 5: Create KING-filtered bfile set (samples filtered in QC'ed dataset)
###############################################################################
log "---- STEP 5: Make KING-filtered dataset (samples only; full QC'ed variants) ----"
log "Command:"
log "  ${PLINK2} --bfile ${STEP2_OUT} --keep ${KING_IN_ID} --make-bed --out ${KING_FILTERED_OUT}"

"${PLINK2}" \
  --bfile "${STEP2_OUT}" \
  --keep "${KING_IN_ID}" \
  --make-bed \
  --out "${KING_FILTERED_OUT}"

KING_FILTERED_SAMPLES=$(count_fam "${KING_FILTERED_OUT}.fam")
KING_FILTERED_VARIANTS=$(count_bim "${KING_FILTERED_OUT}.bim")

log "STEP 5 results (this is your #2 requested bfile set):"
log "  Output prefix:              ${KING_FILTERED_OUT}"
log "  Samples saved (fam lines):  ${KING_FILTERED_SAMPLES}"
log "  Variants saved (bim lines): ${KING_FILTERED_VARIANTS} (should match STEP 2 variants)"
echo "" >> "${LOGFILE}"

###############################################################################
# STEP 6: Create LD-pruned + KING-filtered bfile set (samples + variants)
###############################################################################
log "---- STEP 6: Make LD-pruned + KING-filtered dataset (samples + variants) ----"
log "Command:"
log "  ${PLINK2} --bfile ${STEP2_OUT} --keep ${KING_IN_ID} --extract ${PRUNE_IN} --make-bed --out ${KING_LDPRUNED_OUT}"

"${PLINK2}" \
  --bfile "${STEP2_OUT}" \
  --keep "${KING_IN_ID}" \
  --extract "${PRUNE_IN}" \
  --make-bed \
  --out "${KING_LDPRUNED_OUT}"

KING_LD_SAMPLES=$(count_fam "${KING_LDPRUNED_OUT}.fam")
KING_LD_VARIANTS=$(count_bim "${KING_LDPRUNED_OUT}.bim")

log "STEP 6 results (this is your #3 requested bfile set):"
log "  Output prefix:              ${KING_LDPRUNED_OUT}"
log "  Samples saved (fam lines):  ${KING_LD_SAMPLES}"
log "  Variants saved (bim lines): ${KING_LD_VARIANTS} (should match prune.in count)"
echo "" >> "${LOGFILE}"

###############################################################################
# Final summary block
###############################################################################
log "==== SUMMARY (variants/samples saved + removed/useful) ===="
log "STEP 1 (QC autosomes):"
log "  saved variants: ${STEP1_VARIANTS_OUT} | removed geno: ${STEP1_GENO_REMOVED:-NA} | removed maf: ${STEP1_MAF_REMOVED:-NA}"
log "STEP 2 (QC + keep phenos + SNPs-only):"
log "  saved samples:  ${STEP2_SAMPLES_OUT} | saved variants: ${STEP2_VARIANTS_OUT}"
log "STEP 3 (LD prune):"
log "  useful variants kept (prune.in): ${PRUNE_IN_N} | removed (prune.out): ${PRUNE_OUT_N}"
log "STEP 4 (KING):"
log "  useful variants used: ${KING_VARIANTS_USED:-NA} | unrelated samples kept: ${UNREL_N} | excluded: ${REL_EXCLUDED_N}"
log "OUTPUT SET #1 (final QC before LD): ${STEP2_OUT}"
log "OUTPUT SET #2 (KING-filtered samples, full variants): ${KING_FILTERED_OUT}"
log "OUTPUT SET #3 (LD-pruned + KING-filtered): ${KING_LDPRUNED_OUT}"
log "=== END PIPELINE ==="

echo ""
echo "Done. Global summary log: ${LOGFILE}"
echo "BFILE outputs:"
echo "  1) ${STEP2_OUT}.*"
echo "  2) ${KING_FILTERED_OUT}.*"
echo "  3) ${KING_LDPRUNED_OUT}.*"

