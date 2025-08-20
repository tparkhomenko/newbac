/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Custom palette
        primary: {
          DEFAULT: '#354061', // Headings, highlights, active elements
          50: '#eef0f5',
          500: '#354061',
          600: '#354061',
          700: '#2e3854',
        },
        accent1: {
          DEFAULT: '#DC6DCA', // Buttons, important labels, hover states
          500: '#DC6DCA',
          600: '#DC6DCA',
          700: '#c85db6',
        },
        accent2: {
          DEFAULT: '#8AB4EF', // Charts, progress bars, confidence percentages
          500: '#8AB4EF',
          600: '#8AB4EF',
          700: '#729fe8',
        },
        neutral: {
          DEFAULT: '#B0C9F2', // Background accent / panels, cards, dividers
          50: '#f2f6fd',
          100: '#e6eefb',
          200: '#d3e3fb',
          300: '#B0C9F2',
          600: '#B0C9F2'
        }
      }
    },
  },
  plugins: [],
}
