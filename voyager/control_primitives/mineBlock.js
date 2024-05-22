async function mineBlock(bot, name, count = 1) {
    if (typeof name !== "string") {
        throw new Error(`name for mineBlock must be a string`);
    }
    if (typeof count !== "number") {
        throw new Error(`count for mineBlock must be a number`);
    }
    const blockByName = mcData.blocksByName[name];
    if (!blockByName) {
        throw new Error(`No block named ${name}`);
    }
    const findTargets = () => {
        const blocks = bot.findBlocks({
            matching: [blockByName.id],
            maxDistance: 32,
            count: count,
        });
        const targets = blocks.map(blockPos => bot.blockAt(blockPos));
        targets.forEach(target => console.log(`Found ${name} at ${target.position}`));
        return targets;
    };

    let targets = findTargets();
    if (targets.length === 0) {
        bot.chat(`No ${name} nearby, please explore first`);
        _mineBlockFailCount++;
        if (_mineBlockFailCount > 10) {
            throw new Error(
                "mineBlock failed too many times, make sure you explore before calling mineBlock"
            );
        }
        return;
    }

    let retries = 3;
    while (retries > 0) {
        try {
            await bot.collectBlock.collect(targets, {
                ignoreNoPath: true,
                count: count
            });
            bot.chat(`Mined ${targets.length} ${name}`);
            bot.save(`${name}_mined`);
            break;
        } catch (err) {
            console.error(`Failed to mine blocks: ${err.message}`);
            retries--;
            if (retries === 0) {
                throw new Error(`Failed to mine blocks after multiple attempts: ${err.message}`);
            }
            console.log(`Retrying to mine a block... Attempts left: ${retries}`);
            targets = findTargets();
        }
    }
}
