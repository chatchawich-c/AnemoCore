#!/usr/bin/env bash
set -euo pipefail
ROOT=$(pwd)
# Replace cfg attributes
find "$ROOT" -type f -name "*.rs" -print0 | while IFS= read -r -d '' file; do
  sed -i '' -e 's/#[[[:space:]]]*\[cfg(feature = "rayon"\)]/#[cfg(feature = "std")]/g' \
         -e 's/#[[[:space:]]]*\[cfg(not(feature = "rayon"\)]/#[cfg(not(feature = "std")]/g' "$file"
done
# Insert std imports where Vec or String used without std import
find "$ROOT" -type f -name "*.rs" -print0 | while IFS= read -r -d '' file; do
  if grep -q "Vec<" "$file" || grep -q "String" "$file"; then
    # Insert after the last use line
    line=$(grep -n "^use" "$file" | tail -1 | cut -d: -f1)
    if [ -n "$line" ]; then
      next=$((line+1))
      sed -i '' "${next}i\
#[cfg(feature = "std")] use std::{vec::Vec, string::String};" "$file"
    else
      # Insert at top after #![no_std]
      sed -i '' '1a\
#[cfg(feature = "std")] use std::{vec::Vec, string::String};' "$file"
    fi
  fi
done
# Guard structs and impl blocks for TabWindClimate, CfdSpeedupGrid, TurbineSpec, etc.
for name in TabWindClimate CfdSpeedupGrid TurbineSpec; do
  find "$ROOT" -type f -name "*.rs" -exec sed -i '' "/pub struct $name/,/}/ s/^/#[cfg(feature = \"std\")]\n/" {} +
  find "$ROOT" -type f -name "*.rs" -exec sed -i '' "/impl $name/,/}/ s/^/#[cfg(feature = \"std\")]\n/" {} +
done
# Replace float methods with libm equivalents
find "$ROOT" -type f -name "*.rs" -print0 | while IFS= read -r -d '' file; do
  sed -i '' -e 's/\.powi(\([0-9]\+\))/\n    libm::pow(\0, \1 as f64)/g' "$file"
  sed -i '' -e 's/\.floor()/libm::floor(&0)/g' "$file"
  sed -i '' -e 's/\.sqrt()/libm::sqrt(&0)/g' "$file"
  sed -i '' -e 's/\.exp()/libm::exp(&0)/g' "$file"
  sed -i '' -e 's/\.sin()/libm::sin(&0)/g' "$file"
  sed -i '' -e 's/\.cos()/libm::cos(&0)/g' "$file"
  sed -i '' -e 's/\.log()/libm::log(&0)/g' "$file"
done

echo "Fixes applied"
