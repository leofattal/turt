import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/**", "coverage/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // Backward closures in the math engine capture the parent tensor as a
      // named alias (`a`, `self`) for symmetry with the second operand.
      "@typescript-eslint/no-this-alias": ["error", { allowedNames: ["a", "self"] }],
    },
  },
);
