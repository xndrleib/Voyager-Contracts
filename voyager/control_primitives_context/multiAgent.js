// Send a signal to another player
async function sendSignal(bot) {
  bot.chat("[player signal]")
}

// The waitSignal function listens for a "[signal received]" message to coordinate turn-taking, optionally running a task while waiting
async function waitSignal(bot, task = null, timeoutDuration = 30000) {
    let timeout;

    const chatListening = new Promise((resolve, reject) => {
        function chatHandler(username, message) {
            if (username !== bot.username && message === '[player signal]') {
                bot.chat('[signal received]')
                clearTimeout(timeout);
                resolve();
                bot.removeListener('chat', chatHandler);
            }
        }

        bot.on('chat', chatHandler);
        bot.chat("[waiting signal]")

        timeout = setTimeout(() => {
            bot.removeListener('chat', chatHandler);
            bot.chat("[signal timeout]");
            resolve()
        }, timeoutDuration);
    });

    let taskExecution;
    if (task) {
        taskExecution = task(bot);
        await Promise.all([chatListening, taskExecution]);
    } else {
        await chatListening;
    }
}

// Example of Usage: Synchronized Task Execution
// In this scenario, two bots perform a series of tasks that need to be executed in a specific sequence. Each bot must wait for a signal from the other bot indicating that it has completed its task before starting its own.

// Function for Task 1
async function task1(bot) {
    console.log("Starting task1...");
    // TASK CODE
    sendSignal(bot); // Send signal after task is complete
}

// Function for Task 2
async function task2(bot) {
    console.log("Starting task2...");
    // TASK CODE
    sendSignal(bot); // Send signal after task is complete
}

// Bot 1 code
async function bot1(bot) {
    await waitSignal(bot, task1); // Wait for signal and execute Task 1
    await waitSignal(bot, task2); // Wait for signal and execute Task 2
}

// Bot 2 code
async function bot2(bot) {
    await waitSignal(bot, task2); // Wait for signal and execute Task 2
    await waitSignal(bot, task1); // Wait for signal and execute Task 1
}