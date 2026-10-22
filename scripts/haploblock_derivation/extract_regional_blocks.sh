#!/usr/bin/env bash
###############################################################################
# extract_regional_blocks.sh
#
# Extract regional genotype subsets, detect LD haplotype blocks per region,
# and export per-block .raw files using MINOR-ALLELE dosage.
#
# Final .raw coding:
#   0 = common/common
#   1 = heterozygous
#   2 = minor/minor
#
# Manifest columns required:
#   region_id, group, chr, start_bed, end_bed
# Optional:
#   description
#
# Output layout:
#   preprocessed_blocks/
#   ├── raw/                         # final model-ready .raw files, all together
#   ├── summary/
#   ├── logs/
#   └── regions/<region_id>/
#       ├── qc/                       # allele-frequency and minor-allele audit files
#       ├── snplists/                 # per-haploblock SNP lists
#       └── logs/
###############################################################################

set -euo pipefail

###############################################################################
# Configuration
###############################################################################

# --- Tools ---
PLINK2="${PLINK2:-$(pwd)/bin/plink2}"
PLINK19="${PLINK19:-$(pwd)/bin/plink}"

# --- Inputs ---
PLINK_BFILE="${PLINK_BFILE:-$(pwd)/data/GACRS/raw_chip/garcs_qc_autosomes_qced_king_unrel}"
REGION_MANIFEST="${REGION_MANIFEST:-./data/regions_input_manual.csv}"

# --- Root outputs ---
REGIONAL_OUTPUT_DIR="${REGIONAL_OUTPUT_DIR:-$(pwd)/GACRS/preprocessed_blocks/}"
SUMMARY_DIR="${SUMMARY_DIR:-${REGIONAL_OUTPUT_DIR}/summary}"
LOG_DIR="${LOG_DIR:-${REGIONAL_OUTPUT_DIR}/logs}"
REGIONS_DIR="${REGIONS_DIR:-${REGIONAL_OUTPUT_DIR}/regions}"

REGIONAL_SUMMARY_FILE="${REGIONAL_SUMMARY_FILE:-${SUMMARY_DIR}/regional_summary.csv}"
BLOCK_RAW_MANIFEST="${BLOCK_RAW_MANIFEST:-${SUMMARY_DIR}/block_raw_manifest.tsv}"
CROSS_LD_REPORT="${CROSS_LD_REPORT:-${SUMMARY_DIR}/cross_block_ld.tsv}"
INDEX_SNPS_FILE="${INDEX_SNPS_FILE:-${SUMMARY_DIR}/index_snps_for_cross_ld.txt}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/extract_regional_blocks.log}"

RAW_OUTPUT_DIR="${RAW_OUTPUT_DIR:-$(pwd)/GACRS/preprocessed_blocks/raw}"
RAW_MANIFEST_FILE="${RAW_MANIFEST_FILE:-${RAW_OUTPUT_DIR}/raw_manifest.tsv}"

mkdir -p "${SUMMARY_DIR}" "${LOG_DIR}" "${REGIONS_DIR}"
mkdir -p "${RAW_OUTPUT_DIR}"
mkdir -p "$(dirname "${RAW_MANIFEST_FILE}")"

# --- Extraction QC filters ---
# If upstream final_qc.sh already applied these, set APPLY_EXTRACTION_QC=0.
APPLY_EXTRACTION_QC="${APPLY_EXTRACTION_QC:-1}"
MAF_THRESHOLD="${MAF_THRESHOLD:-0.05}"
GENO_THRESHOLD="${GENO_THRESHOLD:-0.05}"
HWE_THRESHOLD="${HWE_THRESHOLD:-1e-6}"

# --- Raw export ---
EXPORT_MINOR_RAW="${EXPORT_MINOR_RAW:-1}"

# --- LD block detection mode ---
# Options: tas, plink_default
BLOCK_MODE="${BLOCK_MODE:-tas}"

# Tas et al.-style relaxed Gabriel parameters
TAS_BLOCKS_STRONG_LOWCI="${TAS_BLOCKS_STRONG_LOWCI:-0.5005}"
TAS_BLOCKS_STRONG_HIGHCI="${TAS_BLOCKS_STRONG_HIGHCI:-0.8505}"
TAS_BLOCKS_RECOMB_HIGHCI="${TAS_BLOCKS_RECOMB_HIGHCI:-0.70}"
TAS_BLOCKS_MAX_KB="${TAS_BLOCKS_MAX_KB:-10000}"
TAS_BLOCKS_MIN_MAF="${TAS_BLOCKS_MIN_MAF:-0.05}"

# PLINK1.9 default-like Haploview parameters
PLINK_BLOCKS_STRONG_LOWCI="${PLINK_BLOCKS_STRONG_LOWCI:-0.7005}"
PLINK_BLOCKS_STRONG_HIGHCI="${PLINK_BLOCKS_STRONG_HIGHCI:-0.8925}"
PLINK_BLOCKS_RECOMB_HIGHCI="${PLINK_BLOCKS_RECOMB_HIGHCI:-0.90}"
PLINK_BLOCKS_MAX_KB="${PLINK_BLOCKS_MAX_KB:-200}"
PLINK_BLOCKS_MIN_MAF="${PLINK_BLOCKS_MIN_MAF:-0.05}"

# --- Cross-block LD check ---
RUN_CROSS_BLOCK_LD="${RUN_CROSS_BLOCK_LD:-1}"
CROSS_BLOCK_R2="${CROSS_BLOCK_R2:-0.1}"
INDEX_PER_BLOCK="${INDEX_PER_BLOCK:-5}"

# --- Tiny-block pre-merge + final raw-size labeling ---
# Any haploblock SNP list with fewer than this many SNPs is merged into a
# neighboring haploblock before raw export. If only one block remains, it is kept.
MIN_SNPS_PER_BLOCK="${MIN_SNPS_PER_BLOCK:-8}"

# Size class is based on final exported raw variant count.
# small: <50, medium: 50-149, large: >=150
RAW_SMALL_MAX="${RAW_SMALL_MAX:-49}"
RAW_MEDIUM_MAX="${RAW_MEDIUM_MAX:-149}"

###############################################################################
# Resolve active block parameters
###############################################################################

case "${BLOCK_MODE}" in
  tas)
    BLOCKS_STRONG_LOWCI="${TAS_BLOCKS_STRONG_LOWCI}"
    BLOCKS_STRONG_HIGHCI="${TAS_BLOCKS_STRONG_HIGHCI}"
    BLOCKS_RECOMB_HIGHCI="${TAS_BLOCKS_RECOMB_HIGHCI}"
    BLOCKS_MAX_KB="${TAS_BLOCKS_MAX_KB}"
    BLOCKS_MIN_MAF="${TAS_BLOCKS_MIN_MAF}"
    ;;
  plink_default)
    BLOCKS_STRONG_LOWCI="${PLINK_BLOCKS_STRONG_LOWCI}"
    BLOCKS_STRONG_HIGHCI="${PLINK_BLOCKS_STRONG_HIGHCI}"
    BLOCKS_RECOMB_HIGHCI="${PLINK_BLOCKS_RECOMB_HIGHCI}"
    BLOCKS_MAX_KB="${PLINK_BLOCKS_MAX_KB}"
    BLOCKS_MIN_MAF="${PLINK_BLOCKS_MIN_MAF}"
    ;;
  *)
    echo "ERROR: BLOCK_MODE must be 'tas' or 'plink_default', got '${BLOCK_MODE}'"
    exit 1
    ;;
esac

###############################################################################
# Helpers
###############################################################################

: > "${RUN_LOG}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${RUN_LOG}"
}

on_exit() {
  local exit_code=$?

  if [[ "${exit_code}" -eq 0 ]]; then
    log "SCRIPT_EXIT_CODE=${exit_code} : completed successfully"
  else
    log "SCRIPT_EXIT_CODE=${exit_code} : script exited early or failed"
  fi

  exit "${exit_code}"
}

trap on_exit EXIT

run_cmd() {
  local logfile="$1"
  shift
  mkdir -p "$(dirname "${logfile}")"
  set +e
  "$@" > "${logfile}" 2>&1 < /dev/null
  local rc=$?
  set -e
  return "${rc}"
}

count_lines() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    wc -l < "${f}" | xargs
  else
    echo 0
  fi
}

assign_raw_size_class() {
  local n="$1"
  if [[ "${n}" -le "${RAW_SMALL_MAX}" ]]; then
    echo "small"
  elif [[ "${n}" -le "${RAW_MEDIUM_MAX}" ]]; then
    echo "medium"
  else
    echo "large"
  fi
}

raw_tiny_status_suffix() {
  local n="$1"
  if [[ "${n}" -lt "${MIN_SNPS_PER_BLOCK}" ]]; then
    echo "_LT${MIN_SNPS_PER_BLOCK}_AFTER_QC"
  else
    echo ""
  fi
}

list_region_haploblock_snplists() {
  local region_id="$1"
  local dir="$2"
  ls -1 "${dir}/${region_id}_hb"*.snps 2>/dev/null | sort -V || true
}

premerge_haploblock_snplists_for_region() {
  local region_id="$1"
  local min_snps="$2"
  local dir="$3"
  local prefix="${dir}/${region_id}"

  while true; do
    local tmp_list="${dir}/.${region_id}.hb_files.tmp"
    list_region_haploblock_snplists "${region_id}" "${dir}" > "${tmp_list}"

    local n_files
    n_files=$(wc -l < "${tmp_list}" | xargs)

    # 0/1 file => cannot merge further; keep the single block even if tiny.
    if [[ "${n_files}" -le 1 ]]; then
      rm -f "${tmp_list}"
      break
    fi

    local tiny_idx=0
    local tiny_file=""
    local i=1
    while IFS= read -r f; do
      local c
      c=$(count_lines "${f}")
      if [[ "${c}" -gt 0 ]] && [[ "${c}" -lt "${min_snps}" ]]; then
        tiny_idx="${i}"
        tiny_file="${f}"
        break
      fi
      i=$((i + 1))
    done < "${tmp_list}"

    if [[ "${tiny_idx}" -eq 0 ]]; then
      rm -f "${tmp_list}"
      break
    fi

    local left_idx=$((tiny_idx - 1))
    local right_idx=$((tiny_idx + 1))
    local left_file=""
    local right_file=""
    local cl=999999
    local cr=999999

    if [[ "${left_idx}" -ge 1 ]]; then
      left_file=$(sed -n "${left_idx}p" "${tmp_list}" || true)
      [[ -n "${left_file}" ]] && cl=$(count_lines "${left_file}")
    fi
    if [[ "${right_idx}" -le "${n_files}" ]]; then
      right_file=$(sed -n "${right_idx}p" "${tmp_list}" || true)
      [[ -n "${right_file}" ]] && cr=$(count_lines "${right_file}")
    fi

    local target=""
    if [[ -z "${left_file}" ]]; then
      target="${right_file}"
    elif [[ -z "${right_file}" ]]; then
      target="${left_file}"
    else
      if [[ "${cl}" -le "${cr}" ]]; then
        target="${left_file}"
      else
        target="${right_file}"
      fi
    fi

    local tiny_n target_n
    tiny_n=$(count_lines "${tiny_file}")
    target_n=$(count_lines "${target}")
    log "  PREMERGE ${region_id}: merging $(basename "${tiny_file}") (n=${tiny_n}) into $(basename "${target}") (n=${target_n})"

    cat "${tiny_file}" "${target}" | awk 'NF {print $0}' | sort -u > "${target}.tmp"
    mv "${target}.tmp" "${target}"
    rm -f "${tiny_file}"
    rm -f "${tmp_list}"
  done

  # Renumber hb files to be contiguous after merges.
  local tmpdir="${dir}/.${region_id}_hb_renum_tmp"
  rm -rf "${tmpdir}" 2>/dev/null || true
  mkdir -p "${tmpdir}"

  local renum_list="${dir}/.${region_id}.hb_renum_list.tmp"
  list_region_haploblock_snplists "${region_id}" "${dir}" > "${renum_list}"

  local k=1
  while IFS= read -r f; do
    [[ -f "${f}" ]] || continue
    cp "${f}" "${tmpdir}/${region_id}_hb${k}.snps"
    k=$((k + 1))
  done < "${renum_list}"

  rm -f "${renum_list}"
  rm -f "${prefix}_hb"*.snps 2>/dev/null || true
  cp "${tmpdir}/${region_id}_hb"*.snps "${dir}/" 2>/dev/null || true
  rm -rf "${tmpdir}" 2>/dev/null || true
}

rebuild_blocks_tsv_from_snplists() {
  local region_id="$1"
  local snplist_dir="$2"
  local region_bim="$3"
  local blocks_tsv="$4"

  : > "${blocks_tsv}"
  local hb=1
  local snplist
  while IFS= read -r snplist; do
    [[ -s "${snplist}" ]] || continue
    awk -v hb="${hb}" '
      NR == FNR {keep[$1]=1; next}
      ($2 in keep) {
        if (n == 0 || $4 < min_bp) min_bp = $4
        if (n == 0 || $4 > max_bp) max_bp = $4
        chr = $1
        n += 1
        snps = (snps == "" ? $2 : snps "|" $2)
      }
      END {
        if (n > 0) {
          kb = (max_bp - min_bp + 1) / 1000
          printf "%d\t%s\t%d\t%d\t%.1f\t%d\t%s\n", hb, chr, min_bp, max_bp, kb, n, snps
        }
      }
    ' "${snplist}" "${region_bim}" >> "${blocks_tsv}"
    hb=$((hb + 1))
  done < <(list_region_haploblock_snplists "${region_id}" "${snplist_dir}")
}

sanitize_csv_field() {
  echo "$1" | tr -d '"' | tr ',' ';'
}

get_col_idx() {
  local target="$1"
  local i col
  for i in "${!HEADER_ARR[@]}"; do
    col="${HEADER_ARR[$i]}"
    col="${col//\"/}"
    col="${col// /}"
    col="${col//$'\r'/}"
    if [[ "${col}" == "${target}" ]]; then
      echo "${i}"
      return 0
    fi
  done
  echo "-1"
  return 0
}

###############################################################################
# Startup log
###############################################################################

log "=============================================================="
log "extract_regional_blocks.sh"
log "PLINK2              : ${PLINK2}"
log "PLINK1.9            : ${PLINK19}"
log "PLINK bfile         : ${PLINK_BFILE}"
log "Region manifest     : ${REGION_MANIFEST}"
log "Output directory    : ${REGIONAL_OUTPUT_DIR}"
log "Summary directory   : ${SUMMARY_DIR}"
log "Logs directory      : ${LOG_DIR}"
log "Regions directory   : ${REGIONS_DIR}"
log "Block mode          : ${BLOCK_MODE}"
log "  strong-lowci      : ${BLOCKS_STRONG_LOWCI}"
log "  strong-highci     : ${BLOCKS_STRONG_HIGHCI}"
log "  recomb-highci     : ${BLOCKS_RECOMB_HIGHCI}"
log "  max-kb            : ${BLOCKS_MAX_KB}"
log "  min-maf           : ${BLOCKS_MIN_MAF}"
log "Extraction QC       : enabled=${APPLY_EXTRACTION_QC}, MAF>${MAF_THRESHOLD}, geno<${GENO_THRESHOLD}, HWE>${HWE_THRESHOLD}"
log "Export minor raw    : ${EXPORT_MINOR_RAW}"
log "Cross-block LD      : ${RUN_CROSS_BLOCK_LD}, r2>=${CROSS_BLOCK_R2}, index_per_block=${INDEX_PER_BLOCK}"
log "Tiny-block premerge : min_snps=${MIN_SNPS_PER_BLOCK}"
log "Raw size classes    : small<=${RAW_SMALL_MAX}, medium<=${RAW_MEDIUM_MAX}, large>${RAW_MEDIUM_MAX}"
log "=============================================================="

command -v "${PLINK2}" >/dev/null 2>&1 || { log "ERROR: PLINK2 not found: ${PLINK2}"; exit 1; }
command -v "${PLINK19}" >/dev/null 2>&1 || { log "ERROR: PLINK1.9 not found: ${PLINK19}"; exit 1; }

[[ -f "${PLINK_BFILE}.bed" ]] || { log "ERROR: bfile not found: ${PLINK_BFILE}.bed"; exit 1; }
[[ -f "${PLINK_BFILE}.bim" ]] || { log "ERROR: bfile not found: ${PLINK_BFILE}.bim"; exit 1; }
[[ -f "${PLINK_BFILE}.fam" ]] || { log "ERROR: bfile not found: ${PLINK_BFILE}.fam"; exit 1; }
[[ -f "${REGION_MANIFEST}" ]] || { log "ERROR: manifest not found: ${REGION_MANIFEST}"; exit 1; }

###############################################################################
# Manifest validation
###############################################################################

log ""
log "Validating manifest..."

HEADER=$(head -n 1 "${REGION_MANIFEST}")
IFS=',' read -r -a HEADER_ARR <<< "${HEADER}"

REGION_ID_IDX=$(get_col_idx "region_id")
GROUP_IDX=$(get_col_idx "group")
CHR_IDX=$(get_col_idx "chr")
START_BED_IDX=$(get_col_idx "start_bed")
END_BED_IDX=$(get_col_idx "end_bed")
DESCRIPTION_IDX=$(get_col_idx "description")

[[ "${REGION_ID_IDX}" -ge 0 ]] || { log "ERROR: required column 'region_id' missing from manifest"; exit 1; }
[[ "${GROUP_IDX}" -ge 0 ]] || { log "ERROR: required column 'group' missing from manifest"; exit 1; }
[[ "${CHR_IDX}" -ge 0 ]] || { log "ERROR: required column 'chr' missing from manifest"; exit 1; }
[[ "${START_BED_IDX}" -ge 0 ]] || { log "ERROR: required column 'start_bed' missing from manifest"; exit 1; }
[[ "${END_BED_IDX}" -ge 0 ]] || { log "ERROR: required column 'end_bed' missing from manifest"; exit 1; }

HAS_DESCRIPTION=0
[[ "${DESCRIPTION_IDX}" -ge 0 ]] && HAS_DESCRIPTION=1

VALID_CHRS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
ROW_NUM=0
VALIDATION_ERRORS=0

while IFS= read -r line || [[ -n "${line}" ]]; do
  ROW_NUM=$((ROW_NUM + 1))
  [[ ${ROW_NUM} -eq 1 ]] && continue
  [[ -z "${line// /}" ]] && continue

  IFS=',' read -r -a FIELDS <<< "${line}"

  rid="${FIELDS[${REGION_ID_IDX}]}"
  chr="${FIELDS[${CHR_IDX}]}"
  b="${FIELDS[${START_BED_IDX}]}"
  eb="${FIELDS[${END_BED_IDX}]}"

  rid="${rid//\"/}"; rid="${rid// /}"
  chr="${chr//\"/}"; chr="${chr// /}"; chr="${chr#chr}"
  b="${b//\"/}"; b="${b// /}"
  eb="${eb//\"/}"; eb="${eb// /}"

  if [[ ! " ${VALID_CHRS} " =~ " ${chr} " ]]; then
    log "  VALIDATION ERROR row ${ROW_NUM} (${rid}): invalid chr '${chr}'"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
  fi

  if ! [[ "${b}" =~ ^[0-9]+$ ]] || ! [[ "${eb}" =~ ^[0-9]+$ ]]; then
    log "  VALIDATION ERROR row ${ROW_NUM} (${rid}): non-numeric coordinates"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
    continue
  fi

  if [[ ${eb} -le ${b} ]]; then
    log "  VALIDATION ERROR row ${ROW_NUM} (${rid}): end_bed (${eb}) <= start_bed (${b})"
    VALIDATION_ERRORS=$((VALIDATION_ERRORS + 1))
  fi

done < "${REGION_MANIFEST}"

if [[ ${VALIDATION_ERRORS} -gt 0 ]]; then
  log "ERROR: ${VALIDATION_ERRORS} manifest validation error(s); aborting"
  exit 1
fi

log "Manifest validation passed."

###############################################################################
# Initialize output tables
###############################################################################

cat > "${REGIONAL_SUMMARY_FILE}" <<EOF_SUMMARY
region_id,group,description,chr,start_bed,end_bed,size_kb,n_variants_extracted,n_samples,n_blocks,n_snps_in_blocks,n_unassigned_snps,pct_unassigned_snps,unassigned_snps_file,mean_block_kb,max_block_kb,mean_block_snps,max_block_snps,used_fallback,block_formation_status,n_raw_blocks_exported
EOF_SUMMARY

cat > "${BLOCK_RAW_MANIFEST}" <<EOF_MANIFEST
haploblock_id region_id group chr bp1 bp2 kb n_snps out_prefix raw_file minor_alleles_qc minor_alleles_file freq_file n_raw_variants block_size_class status
EOF_MANIFEST

echo -e "haploblock_id\tregion_id\tgroup\traw_file\tn_raw_variants\tblock_size_class\tstatus" > "${RAW_MANIFEST_FILE}"

: > "${INDEX_SNPS_FILE}"

###############################################################################
# Per-region processing
###############################################################################

ROW_NUM=0
N_PROCESSED=0
N_FAILED=0
N_FALLBACK=0
N_RAW_BLOCKS=0
N_RAW_SMALL=0
N_RAW_MEDIUM=0
N_RAW_LARGE=0
N_RAW_LT_MIN_AFTER_QC=0

while IFS= read -r line || [[ -n "${line}" ]]; do
  ROW_NUM=$((ROW_NUM + 1))
  [[ ${ROW_NUM} -eq 1 ]] && continue
  [[ -z "${line// /}" ]] && continue

  IFS=',' read -r -a FIELDS <<< "${line}"

  region_id="${FIELDS[${REGION_ID_IDX}]}"
  group="${FIELDS[${GROUP_IDX}]}"
  chr="${FIELDS[${CHR_IDX}]}"
  start_bed="${FIELDS[${START_BED_IDX}]}"
  end_bed="${FIELDS[${END_BED_IDX}]}"
  description=""
  [[ ${HAS_DESCRIPTION} -eq 1 ]] && description="${FIELDS[${DESCRIPTION_IDX}]}"

  region_id="${region_id//\"/}"; region_id="${region_id// /}"
  group="${group//\"/}"; group="${group// /}"
  chr="${chr//\"/}"; chr="${chr// /}"; chr="${chr#chr}"
  start_bed="${start_bed//\"/}"; start_bed="${start_bed// /}"
  end_bed="${end_bed//\"/}"; end_bed="${end_bed// /}"
  description=$(sanitize_csv_field "${description}")

  from_bp=$((start_bed + 1))
  to_bp="${end_bed}"
  size_kb=$(awk -v s="${start_bed}" -v e="${end_bed}" 'BEGIN { printf "%.1f", (e-s)/1000 }')

  region_dir="${REGIONS_DIR}/${region_id}"
  qc_dir="${region_dir}/qc"
  snplist_dir="${region_dir}/snplists"
  region_log_dir="${region_dir}/logs"
  mkdir -p "${qc_dir}" "${snplist_dir}" "${region_log_dir}" "${RAW_OUTPUT_DIR}"

  region_qc_prefix="${qc_dir}/${region_id}"
  region_block_prefix="${region_dir}/${region_id}"
  region_log="${region_log_dir}/${region_id}.extract_region.log"

  log ""
  log "[$((ROW_NUM-1))] ${region_id} (${group}) chr${chr}:${from_bp}-${to_bp} size=${size_kb} kb"

  used_fallback=0
  status="UNKNOWN"
  n_raw_blocks_exported=0

  #############################################################################
  # Step 1: Extract region with PLINK2
  #############################################################################

  extract_cmd=(
    "${PLINK2}"
    --bfile "${PLINK_BFILE}"
    --allow-extra-chr
    --no-psam-pheno
    --chr "${chr}"
    --from-bp "${from_bp}"
    --to-bp "${to_bp}"
    --max-alleles 2
    --snps-only just-acgt
    --set-all-var-ids '@:#:$1:$2'
    --make-founders
    --make-bed
    --out "${region_qc_prefix}"
  )

  if [[ "${APPLY_EXTRACTION_QC}" -eq 1 ]]; then
    extract_cmd+=(--maf "${MAF_THRESHOLD}" --geno "${GENO_THRESHOLD}" --hwe "${HWE_THRESHOLD}")
  fi

  if ! run_cmd "${region_log}" "${extract_cmd[@]}"; then
    log "  ERROR: extraction failed; see ${region_log}"
    N_FAILED=$((N_FAILED + 1))
    echo "${region_id},${group},${description},${chr},${start_bed},${end_bed},${size_kb},0,0,0,0,0,NA,NA,NA,NA,NA,NA,0,FAILED_EXTRACTION,0" >> "${REGIONAL_SUMMARY_FILE}"
    continue
  fi

  if [[ ! -f "${region_qc_prefix}.bim" ]]; then
    log "  ERROR: no .bim produced"
    N_FAILED=$((N_FAILED + 1))
    echo "${region_id},${group},${description},${chr},${start_bed},${end_bed},${size_kb},0,0,0,0,0,NA,NA,NA,NA,NA,NA,0,NO_BIM_AFTER_EXTRACTION,0" >> "${REGIONAL_SUMMARY_FILE}"
    continue
  fi

  n_variants=$(count_lines "${region_qc_prefix}.bim")
  n_samples=$(count_lines "${region_qc_prefix}.fam")

  if [[ "${n_variants}" -eq 0 ]]; then
    log "  ERROR: no variants survived extraction/QC"
    N_FAILED=$((N_FAILED + 1))
    echo "${region_id},${group},${description},${chr},${start_bed},${end_bed},${size_kb},0,${n_samples},0,0,0,NA,NA,NA,NA,NA,NA,0,NO_VARIANTS_AFTER_QC,0" >> "${REGIONAL_SUMMARY_FILE}"
    continue
  fi

  log "  Extracted: ${n_variants} variants, ${n_samples} samples"

  awk '{print $2}' "${region_qc_prefix}.bim" > "${qc_dir}/${region_id}.snplist"

  #############################################################################
  # Step 2: LD block detection with PLINK1.9
  #############################################################################

  if [[ "${n_variants}" -lt 2 ]]; then
    log "  Only ${n_variants} variant(s); skipping block detection, using fallback"
    n_blocks=0
  else
    if ! run_cmd "${region_log_dir}/${region_id}.plink19_blocks.log" \
      "${PLINK19}" \
        --bfile "${region_qc_prefix}" \
        --blocks no-pheno-req \
        --blocks-strong-lowci "${BLOCKS_STRONG_LOWCI}" \
        --blocks-strong-highci "${BLOCKS_STRONG_HIGHCI}" \
        --blocks-recomb-highci "${BLOCKS_RECOMB_HIGHCI}" \
        --blocks-max-kb "${BLOCKS_MAX_KB}" \
        --blocks-min-maf "${BLOCKS_MIN_MAF}" \
        --out "${region_block_prefix}"
    then
      log "  WARNING: block detection failed; using fallback"
      n_blocks=0
    elif [[ -f "${region_block_prefix}.blocks.det" ]]; then
      n_blocks=$(awk 'END {print NR-1}' "${region_block_prefix}.blocks.det")
      [[ "${n_blocks}" -lt 0 ]] && n_blocks=0
    else
      n_blocks=0
    fi
  fi

  #############################################################################
  # Step 3: Fallback single block if none detected
  #############################################################################

  if [[ "${n_blocks}" -eq 0 ]]; then
    used_fallback=1
    N_FALLBACK=$((N_FALLBACK + 1))
    log "  No blocks detected; writing fallback single-block covering full region"

    snps_str=$(awk '{print $2}' "${region_qc_prefix}.bim" | paste -sd '|')
    {
      echo -e " CHR\tBP1\tBP2\tKB\tNSNPS\tSNPS"
      echo -e " ${chr}\t${from_bp}\t${to_bp}\t${size_kb}\t${n_variants}\t${snps_str}"
    } > "${region_block_prefix}.blocks.det"

    echo "* ${snps_str//|/ }" > "${region_block_prefix}.blocks"
    n_blocks=1
  fi

  #############################################################################
  # Step 4: Block statistics and unassigned SNP audit
  #############################################################################

  if [[ -f "${region_block_prefix}.blocks.det" ]] && [[ "${n_blocks}" -gt 0 ]]; then
    block_stats=$(awk '
      NR > 1 {
        kb=$4
        nsnps=$5
        total_snps += nsnps
        sum_kb += kb
        if (kb > max_kb) max_kb = kb
        if (nsnps > max_snps) max_snps = nsnps
        n++
      }
      END {
        if (n > 0) {
          printf "%d %.1f %.1f %.1f %d", total_snps, sum_kb/n, max_kb, total_snps/n, max_snps
        } else {
          printf "0 NA NA NA NA"
        }
      }' "${region_block_prefix}.blocks.det")

    read -r n_snps_in_blocks mean_block_kb max_block_kb mean_block_snps max_block_snps <<< "${block_stats}"
  else
    n_snps_in_blocks=0
    mean_block_kb="NA"
    max_block_kb="NA"
    mean_block_snps="NA"
    max_block_snps="NA"
  fi

  # Unassigned SNP audit: all extracted variants minus all SNPs present in blocks.det.
  all_snps_sorted="${qc_dir}/${region_id}.all_snps.sorted.txt"
  assigned_snps_sorted="${qc_dir}/${region_id}.assigned_snps.sorted.txt"
  unassigned_snps="${qc_dir}/${region_id}.unassigned_snps.txt"

  awk '{print $2}' "${region_qc_prefix}.bim" | sort -u > "${all_snps_sorted}"
  awk 'NR > 1 {gsub(/\|/, "\n", $6); print $6}' "${region_block_prefix}.blocks.det" | awk 'NF {print $1}' | sort -u > "${assigned_snps_sorted}"
  comm -23 "${all_snps_sorted}" "${assigned_snps_sorted}" > "${unassigned_snps}" || true

  n_unassigned_snps=$(count_lines "${unassigned_snps}")
  pct_unassigned_snps=$(awk -v u="${n_unassigned_snps}" -v n="${n_variants}" 'BEGIN { if (n > 0) printf "%.2f", 100*u/n; else print "NA" }')

  if [[ "${used_fallback}" -eq 1 ]]; then
    status="FALLBACK_SINGLE_BLOCK"
  elif [[ "${n_blocks}" -eq 1 ]]; then
    status="OK_SINGLE_BLOCK"
  elif [[ "${n_blocks}" -ge 2 ]]; then
    status="OK_MULTI_BLOCK"
  else
    status="NO_BLOCKS"
  fi

  log "  Blocks: ${n_blocks}, SNPs-in-blocks: ${n_snps_in_blocks}, unassigned: ${n_unassigned_snps} (${pct_unassigned_snps}%), mean_kb: ${mean_block_kb}, max_kb: ${max_block_kb}, status: ${status}"

  #############################################################################
  # Step 5: Export each LD block as minor-allele-dosage .raw
  #
  # Naming:
  #   We use "hb" for haploblock IDs/files:
  #     region_9p24_IL33_hb1.raw
  #     region_9p24_IL33_hb1.snps
  #
  # Important:
  #   PLINK1.9 .blocks.det is whitespace-aligned, not guaranteed tab-delimited.
  #   Do NOT parse it with `IFS=$'\t' read ...` directly.
  #   First normalize it with awk and write one clean .snps file per haploblock.
  #############################################################################

  if [[ "${EXPORT_MINOR_RAW}" -eq 1 ]]; then
    rm -f "${snplist_dir}/${region_id}_hb"*.snps 2>/dev/null || true

    blocks_tsv="${region_block_prefix}.blocks.tsv"
    : > "${blocks_tsv}"

    awk -v pref="${snplist_dir}/${region_id}" '
      NR == 1 {next}
      NF >= 6 {
        hb += 1
        fn = pref "_hb" hb ".snps"
        snps = $6
        gsub(/\|/, "\n", snps)
        print snps > fn
        close(fn)
        print hb "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6
      }
    ' "${region_block_prefix}.blocks.det" > "${blocks_tsv}"

    if [[ ! -s "${blocks_tsv}" ]]; then
      log "  WARNING: ${region_id}: no parseable haploblocks found in ${region_block_prefix}.blocks.det"
    else
      log "  Pre-merging haploblocks with <${MIN_SNPS_PER_BLOCK} SNPs before raw export"
      premerge_haploblock_snplists_for_region "${region_id}" "${MIN_SNPS_PER_BLOCK}" "${snplist_dir}"
      rebuild_blocks_tsv_from_snplists "${region_id}" "${snplist_dir}" "${region_qc_prefix}.bim" "${blocks_tsv}"
      n_blocks=$(awk 'END {print NR}' "${blocks_tsv}")

      # Refresh regional block statistics so regional_summary.csv matches the
      # actual pre-merged haploblocks used for raw export.
      premerge_block_stats=$(awk '
        {
          kb=$5
          nsnps=$6
          total_snps += nsnps
          sum_kb += kb
          if (kb > max_kb) max_kb = kb
          if (nsnps > max_snps) max_snps = nsnps
          n++
        }
        END {
          if (n > 0) {
            printf "%d %.1f %.1f %.1f %d", total_snps, sum_kb/n, max_kb, total_snps/n, max_snps
          } else {
            printf "0 NA NA NA NA"
          }
        }' "${blocks_tsv}")
      read -r n_snps_in_blocks mean_block_kb max_block_kb mean_block_snps max_block_snps <<< "${premerge_block_stats}"

      if [[ "${used_fallback}" -eq 1 ]]; then
        status="FALLBACK_SINGLE_BLOCK"
      elif [[ "${n_blocks}" -eq 1 ]]; then
        status="OK_SINGLE_BLOCK"
      else
        status="OK_MULTI_BLOCK_PREMERGED"
      fi

      log "  Haploblocks after premerge: ${n_blocks}"
    fi

    while IFS=$'\t' read -r hb b_chr b_bp1 b_bp2 b_kb b_nsnps b_snps; do
      [[ -z "${hb// /}" ]] && continue

      block_id="${region_id}_hb${hb}"

      # Region-local prefix for temporary files, logs, frequency, and allele QC.
      block_work_prefix="${qc_dir}/${block_id}"

      # Global prefix for final model-ready .raw files.
      raw_prefix="${RAW_OUTPUT_DIR}/${block_id}"

      block_snplist="${snplist_dir}/${block_id}.snps"
      tmp_block="${block_work_prefix}.tmp_qc"
      tmp_freq="${block_work_prefix}.freq_for_minor"
      minor_alleles="${block_work_prefix}.minor_alleles.txt"
      minor_qc="${block_work_prefix}.minor_alleles_qc.tsv"

      if [[ ! -s "${block_snplist}" ]]; then
        log "  WARNING: ${block_id}: empty haploblock SNP list; skipping"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\tNA\tNA\tNA\t0\tNA\tEMPTY_BLOCK_SNPLIST" >> "${BLOCK_RAW_MANIFEST}"
        continue
      fi

      if ! run_cmd "${region_log_dir}/${block_id}.extract.log" \
        "${PLINK2}" \
          --bfile "${region_qc_prefix}" \
          --extract "${block_snplist}" \
          --make-bed \
          --out "${tmp_block}"
      then
        log "  WARNING: ${block_id}: haploblock extraction failed"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\tNA\tNA\tNA\t0\tNA\tFAILED_BLOCK_EXTRACTION" >> "${BLOCK_RAW_MANIFEST}"
        rm -f "${tmp_block}".* 2>/dev/null || true
        continue
      fi

      n_block_vars=$(count_lines "${tmp_block}.bim")
      if [[ "${n_block_vars}" -eq 0 ]]; then
        log "  WARNING: ${block_id}: 0 variants after haploblock extraction"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\tNA\tNA\tNA\t0\tNA\tNO_VARIANTS_IN_BLOCK" >> "${BLOCK_RAW_MANIFEST}"
        rm -f "${tmp_block}".* 2>/dev/null || true
        continue
      fi

      if ! run_cmd "${region_log_dir}/${block_id}.freq_for_minor.log" \
        "${PLINK2}" \
          --bfile "${tmp_block}" \
          --freq \
          --out "${tmp_freq}"
      then
        log "  WARNING: ${block_id}: frequency calculation failed"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\tNA\tNA\tNA\t0\tNA\tFAILED_FREQ" >> "${BLOCK_RAW_MANIFEST}"
        rm -f "${tmp_block}".* 2>/dev/null || true
        continue
      fi

      if [[ ! -f "${tmp_freq}.afreq" ]]; then
        log "  WARNING: ${block_id}: missing ${tmp_freq}.afreq"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\tNA\tNA\tNA\t0\tNA\tMISSING_AFREQ" >> "${BLOCK_RAW_MANIFEST}"
        rm -f "${tmp_block}".* 2>/dev/null || true
        continue
      fi

      # PLINK2 .afreq layout:
      #   #CHROM ID REF ALT PROVISIONAL_REF? ALT_FREQS OBS_CT
      #
      # Final .raw dosage is minor-allele dosage:
      #   0 = common/common
      #   1 = heterozygous
      #   2 = minor/minor
      awk '
        BEGIN {
          OFS="\t"
          print "variant_id", "ref", "alt", "alt_freq", "minor_allele", "counted_allele_freq", "action"
        }
        NR > 1 {
          id=$2
          ref=$3
          alt=$4
          alt_freq=$6 + 0
          if (alt_freq <= 0.5) {
            minor=alt
            counted_freq=alt_freq
            action="COUNT_ALT"
          } else {
            minor=ref
            counted_freq=1 - alt_freq
            action="COUNT_REF"
          }
          print id, ref, alt, alt_freq, minor, counted_freq, action
        }
      ' "${tmp_freq}.afreq" > "${minor_qc}"

      awk 'NR > 1 {print $1, $5}' "${minor_qc}" > "${minor_alleles}"

      if [[ ! -s "${minor_alleles}" ]]; then
        log "  WARNING: ${block_id}: empty minor allele file"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\t${minor_qc}\t${minor_alleles}\t${tmp_freq}.afreq\t0\tNA\tEMPTY_MINOR_ALLELES" >> "${BLOCK_RAW_MANIFEST}"
        rm -f "${tmp_block}".* 2>/dev/null || true
        continue
      fi

      if ! run_cmd "${region_log_dir}/${block_id}.export_minor_raw.log" \
        "${PLINK2}" \
          --bfile "${tmp_block}" \
          --export A \
          --export-allele "${minor_alleles}" \
          --out "${raw_prefix}"
      then
        log "  WARNING: ${block_id}: minor-dosage raw export failed"
        echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${b_nsnps}\t${raw_prefix}\tNA\t${minor_qc}\t${minor_alleles}\t${tmp_freq}.afreq\t0\tNA\tFAILED_RAW_EXPORT" >> "${BLOCK_RAW_MANIFEST}"
        rm -f "${tmp_block}".* 2>/dev/null || true
        continue
      fi

      n_raw_variants=0
      if [[ -f "${raw_prefix}.raw" ]]; then
        n_raw_variants=$(head -n 1 "${raw_prefix}.raw" | awk '{print NF - 6}')
      fi

      index_ranked="${block_work_prefix}.index_snps_ranked.tmp"

      awk '
        NR > 1 {
          altf=$6 + 0
          maf=(altf <= 0.5 ? altf : 1-altf)
          print $2, maf
        }
      ' "${tmp_freq}.afreq" > "${index_ranked}"

      sort -k2,2nr "${index_ranked}" > "${index_ranked}.sorted"

      awk -v n="${INDEX_PER_BLOCK}" 'NR <= n {print $1}' "${index_ranked}.sorted" >> "${INDEX_SNPS_FILE}"

      rm -f "${index_ranked}" "${index_ranked}.sorted"

      block_size_class=$(assign_raw_size_class "${n_raw_variants}")
      status_suffix=$(raw_tiny_status_suffix "${n_raw_variants}")
      final_status="ok${status_suffix}"

      if [[ "${n_raw_variants}" -lt "${MIN_SNPS_PER_BLOCK}" ]]; then
        log "  WARNING: ${block_id}: only ${n_raw_variants} raw variants after QC/export (<${MIN_SNPS_PER_BLOCK}); keeping and labeling ${final_status}"
        N_RAW_LT_MIN_AFTER_QC=$((N_RAW_LT_MIN_AFTER_QC + 1))
      fi

      case "${block_size_class}" in
        small)  N_RAW_SMALL=$((N_RAW_SMALL + 1)) ;;
        medium) N_RAW_MEDIUM=$((N_RAW_MEDIUM + 1)) ;;
        large)  N_RAW_LARGE=$((N_RAW_LARGE + 1)) ;;
      esac

      echo -e "${block_id}\t${region_id}\t${group}\t${chr}\t${b_bp1}\t${b_bp2}\t${b_kb}\t${n_block_vars}\t${raw_prefix}\t${raw_prefix}.raw\t${minor_qc}\t${minor_alleles}\t${tmp_freq}.afreq\t${n_raw_variants}\t${block_size_class}\t${final_status}" >> "${BLOCK_RAW_MANIFEST}"

      echo -e "${block_id}\t${region_id}\t${group}\t${raw_prefix}.raw\t${n_raw_variants}\t${block_size_class}\t${final_status}" >> "${RAW_MANIFEST_FILE}"

      n_raw_blocks_exported=$((n_raw_blocks_exported + 1))
      N_RAW_BLOCKS=$((N_RAW_BLOCKS + 1))

      rm -f "${tmp_block}".* 2>/dev/null || true
    done < "${blocks_tsv}"
  fi

  echo "${region_id},${group},${description},${chr},${start_bed},${end_bed},${size_kb},${n_variants},${n_samples},${n_blocks},${n_snps_in_blocks},${n_unassigned_snps},${pct_unassigned_snps},${unassigned_snps},${mean_block_kb},${max_block_kb},${mean_block_snps},${max_block_snps},${used_fallback},${status},${n_raw_blocks_exported}" >> "${REGIONAL_SUMMARY_FILE}"

  N_PROCESSED=$((N_PROCESSED + 1))
done < "${REGION_MANIFEST}"

###############################################################################
# Cross-block LD check
###############################################################################

if [[ "${RUN_CROSS_BLOCK_LD}" -eq 1 ]]; then
  log ""
  log "Running cross-block LD check..."

  sort -u "${INDEX_SNPS_FILE}" -o "${INDEX_SNPS_FILE}" 2>/dev/null || true
  echo -e "snp_a\tsnp_b\tr2\tlevel" > "${CROSS_LD_REPORT}"

  if [[ -s "${INDEX_SNPS_FILE}" ]]; then
    cross_prefix="${SUMMARY_DIR}/cross_block_ld"

    if run_cmd "${LOG_DIR}/cross_block_ld.plink2_extract.log" \
      "${PLINK2}" \
        --bfile "${PLINK_BFILE}" \
        --allow-extra-chr \
        --no-psam-pheno \
        --set-all-var-ids '@:#:$1:$2' \
        --extract "${INDEX_SNPS_FILE}" \
        --make-bed \
        --out "${cross_prefix}.tmp"
    then
      if ! run_cmd "${LOG_DIR}/cross_block_ld.plink2_r2.log" \
        "${PLINK2}" \
          --bfile "${cross_prefix}.tmp" \
          --r2-unphased \
          --ld-window 999999 \
          --ld-window-kb 999999 \
          --ld-window-r2 "${CROSS_BLOCK_R2}" \
          --out "${cross_prefix}"
      then
        log "  WARNING: PLINK2 cross-block LD failed; trying PLINK1.9"
        run_cmd "${LOG_DIR}/cross_block_ld.plink19_r2.log" \
          "${PLINK19}" \
            --bfile "${cross_prefix}.tmp" \
            --r2 \
            --ld-window 999999 \
            --ld-window-kb 999999 \
            --ld-window-r2 "${CROSS_BLOCK_R2}" \
            --out "${cross_prefix}" || true
      fi

      for ld_file in "${cross_prefix}.vcor" "${cross_prefix}.ld"; do
        if [[ -f "${ld_file}" ]]; then
          awk -v thresh="${CROSS_BLOCK_R2}" '
            NR > 1 {
              n=NF
              r2=$n + 0
              snp_a=$(n-2)
              snp_b=$(n-1)
              if (snp_a != snp_b && r2 >= thresh) {
                level=(r2 >= 0.3 ? "HIGH" : "MODERATE")
                print snp_a "\t" snp_b "\t" r2 "\t" level
              }
            }
          ' "${ld_file}" >> "${CROSS_LD_REPORT}"
          break
        fi
      done

      n_cross=$(awk 'NR > 1' "${CROSS_LD_REPORT}" | wc -l | xargs)
      log "  Cross-block LD pairs with r2 >= ${CROSS_BLOCK_R2}: ${n_cross}"
      rm -f "${cross_prefix}.tmp".* 2>/dev/null || true
    else
      log "  WARNING: could not extract index SNPs for cross-block LD; see ${LOG_DIR}/cross_block_ld.plink2_extract.log"
    fi
  else
    log "  No index SNPs found; skipping cross-block LD"
  fi
fi

###############################################################################
# Wrap-up
###############################################################################

log ""
log "=============================================================="
log "Processing complete"
log "  Regions processed       : ${N_PROCESSED}"
log "  Regions used fallback   : ${N_FALLBACK}"
log "  Regions failed          : ${N_FAILED}"
log "  Raw blocks exported     : ${N_RAW_BLOCKS}"
log "  Raw size classes        : small=${N_RAW_SMALL}, medium=${N_RAW_MEDIUM}, large=${N_RAW_LARGE}"
log "  Raw blocks <${MIN_SNPS_PER_BLOCK} after QC/export: ${N_RAW_LT_MIN_AFTER_QC}"
log "  Summary CSV             : ${REGIONAL_SUMMARY_FILE}"
log "  Block/raw manifest      : ${BLOCK_RAW_MANIFEST}"
log "  Cross-block LD report   : ${CROSS_LD_REPORT}"
log "  Raw manifest            : ${RAW_MANIFEST_FILE}"
log "  Raw output directory    : ${RAW_OUTPUT_DIR}"
log "  Logs directory          : ${LOG_DIR}"
log "  Regions directory       : ${REGIONS_DIR}"
log "=============================================================="
