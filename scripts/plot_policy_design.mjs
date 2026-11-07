import {execFileSync} from 'node:child_process';
execFileSync(process.env.PYTHON || 'python',[new URL('./update_presentation_figures.py',import.meta.url).pathname.replace(/^\/([A-Za-z]:)/,'$1')],{stdio:'inherit'});
