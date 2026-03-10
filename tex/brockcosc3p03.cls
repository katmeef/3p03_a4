%% brock-cosc-3p03.cls
%% A basic LaTeX class file with custom margins and headers/footers

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{brockcosc3p03}[2026/01/05 Brock COSC 3P03 Document Class]

% Base class
\LoadClass[12pt, letterpaper]{article}

% Required packages
\RequirePackage[top=1in, bottom=1in, left=1in, right=1in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{graphicx}
\RequirePackage{lastpage}

% Page style setup
\pagestyle{fancy}
\fancyhf{} % Clear default headers and footers

% Header configuration
\fancyhead[L]{COSC 3P03: Winter 2026\\Assignment \assignnum} % Section name on left
\fancyhead[R]{Name: \studentname\\Student ID: \studentid} % Page number on right

% Footer configuration
\fancyfoot[C]{} % Clear center footer
\fancyfoot[L]{\today} % Date on left
\fancyfoot[R]{Page \thepage\ of \pageref{LastPage}} % Page x of y on right

% Header/footer line thickness
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% Adjust plain page style (used on chapter/section first pages)
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

% Custom commands for user-defined header/footer content
\newcommand{\documenttitle}[1]{\def\@documenttitle{#1}}
\newcommand{\documentauthor}[1]{\def\@documentauthor{#1}}

% Optional: Override header with custom title and author
\newcommand{\useCustomHeader}{
  \fancyhead[L]{\@documenttitle}
  \fancyhead[R]{\@documentauthor}
}

\endinput