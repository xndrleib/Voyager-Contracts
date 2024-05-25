await bot.pathfinder.goto(goal); // A very useful function. This function may change your main-hand equipment.
// Following are some Goals you can use:
new GoalNear(x, y, z, range); // Move the bot to a block within the specified range of the specified block. `x`, `y`, `z`, and `range` are `number`
new GoalXZ(x, z); // Useful for long-range goals that don't have a specific Y level. `x` and `z` are `number`
new GoalGetToBlock(x, y, z); // Not get into the block, but get directly adjacent to it. Useful for fishing, farming, filling bucket, and beds. `x`, `y`, and `z` are `number`
new GoalFollow(entity, range); // Follow the specified entity within the specified range. `entity` is `Entity`, `range` is `number`
new GoalPlaceBlock(position, bot.world, {}); // Position the bot in order to place a block. `position` is `Vec3`
new GoalLookAtBlock(position, bot.world, {}); // Path into a position where a blockface of the block at position is visible. `position` is `Vec3`

// These are other Mineflayer functions you can use:
bot.blockAt(position); // Return the block at `position`. `position` is `Vec3`
bot.findBlock(options); // Return the nearest block (not position) matching the specified options. `options` is `FindBlockOptions`. Important: must use block.position to get `Vec3` position for use in other functions

// These are other Mineflayer async functions you can use:
await bot.equip(item, destination); // Equip the item in the specified destination. `item` is `Item`, `destination` can only be "hand", "head", "torso", "legs", "feet", "off-hand"
await bot.activateBlock(block); // This is the same as right-clicking a block in the game. Useful for buttons, doors, etc. You must get to the block first
await bot.lookAt(position); // Look at the specified position. You must go near the position before you look at it. To fill bucket with water, you must lookAt first. `position` is `Vec3`

// node-minecraft-data is loaded as mcData. You can use it to get information about items and block in Minecraft
redMushroomBlock = mcData.blocksByName['red_mushroom_block']; // Object containing information about "Red Mushroom Block"
redMushroomBlockID = redMushroomBlock.id; // ID of "Red Mushroom Block"
wheatItem = mcData.itemsByName['wheat']; // Object containing information about "Wheat" item
wheatItemID = wheatItem.id; // ID of "Wheat" item
