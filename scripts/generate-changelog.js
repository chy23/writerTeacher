import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

// Get the directory of the current module
const __dirname = path.resolve();

function generateChangelog() {
    try {
        // Fetch the last 50 commits
        // Format: Hash|||Date|||Subject|||Body===END_COMMIT===
        const command = `git log -n 50 --pretty=format:"%h|||%cd|||%s|||%b===END_COMMIT===" --date=short`;
        const output = execSync(command, { encoding: 'utf-8' });

        const rawCommits = output.split('===END_COMMIT===');
        
        const changelog = [];
        
        for (const rawCommit of rawCommits) {
            const commitStr = rawCommit.trim();
            if (!commitStr) continue;

            const parts = commitStr.split('|||');
            if (parts.length < 4) continue;

            const hash = parts[0].trim();
            const date = parts[1].trim();
            const subject = parts[2].trim();
            const body = parts[3].trim();

            // Determine if this is a bug fix
            const lowerSubject = subject.toLowerCase();
            const lowerBody = body.toLowerCase();
            const isBugFix = lowerSubject.includes('fix') || lowerSubject.includes('修復') || lowerSubject.includes('解決') ||
                             lowerBody.includes('fix') || lowerBody.includes('修復') || lowerBody.includes('解決');

            // Filter out empty details
            const details = body ? body.split('\n').map(line => line.trim()).filter(line => line.length > 0) : [];

            changelog.push({
                version: `#${hash}`,
                date: date,
                title: subject,
                details: details,
                isBugFix: isBugFix
            });
        }

        // Write to src/changelog.json
        const targetPath = path.join(__dirname, 'src', 'changelog.json');
        fs.writeFileSync(targetPath, JSON.stringify(changelog, null, 2), 'utf-8');
        console.log(`Successfully generated changelog with ${changelog.length} entries.`);
    } catch (error) {
        console.error("Failed to generate changelog:", error.message);
        // Fallback to empty array if git fails (e.g., when not in a git repo)
        const targetPath = path.join(__dirname, 'src', 'changelog.json');
        if (!fs.existsSync(targetPath)) {
            fs.writeFileSync(targetPath, JSON.stringify([], null, 2), 'utf-8');
        }
    }
}

generateChangelog();
