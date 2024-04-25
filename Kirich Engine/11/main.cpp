#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <unordered_map>
#include <random>
#include <thread>

#define SHOW_FPS 1
#define STOP_EVERY_FRAME 0

#define MAX_FRAMES_IN_FLIGHT 2

#define WIDTH 800
#define HEIGHT 800

#define MAX_PARTICLE_COUNT 10000


#define CASE 2

#if CASE == 0
#define PARTICLE_DIVISION 8 // rmin * 4
#define PARTICLE_PARAMETR_COUNT 2
#define PARTICLE_PARAMETRS { {2, 0.2f, 1000}, {1, 1, 10000} }
#elif CASE == 1
#define PARTICLE_DIVISION 8 
#define PARTICLE_PARAMETR_COUNT 2
#define PARTICLE_PARAMETRS { {2, 1, 0}, {2, 4.0f, 1000} }
#elif CASE == 2
#define PARTICLE_DIVISION 8 
#define PARTICLE_PARAMETR_COUNT 2
#define PARTICLE_PARAMETRS { {2, 1, 1000}, {0, 0, 0} }
#else
#define PARTICLE_DIVISION 1 
#define PARTICLE_PARAMETR_COUNT 2
#define PARTICLE_PARAMETRS { {0, 0, 0}, {0, 0, 0} }
#endif 


#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

static inline int Compar(const void* p1, const void* p2) {
	int x1 = *(int*)p1;
	int x2 = *(int*)p2;
	int y1 = *((int*)p1 + 1);
	int y2 = *((int*)p2 + 1);
	int z1 = *((int*)p1 + 2);
	int z2 = *((int*)p2 + 2);
	if (z1 > z2)
		return 1;
	if (z1 < z2)
		return -1;
	if (y1 > y2)
		return 1;
	if (y1 < y2)
		return -1;
	if (x1 > x2)
		return 1;
	if (x1 < x2)
		return -1;
	return 0;
}

struct Particles {
	alignas(16) glm::vec3 position;
	alignas(16) glm::vec3 lposition;
	alignas(4) int type;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Particles);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Particles, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Particles, lposition);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32_SINT;
		attributeDescriptions[2].offset = offsetof(Particles, type);

		return attributeDescriptions;
	}
};

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsAndComputeFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObjectParticleType {
	alignas(4) float rmin;
	alignas(4) float e;
	alignas(8) float m;

};

struct UniformBufferObject {
	UniformBufferObjectParticleType particleParametrs[PARTICLE_PARAMETR_COUNT] = PARTICLE_PARAMETRS;
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
	alignas(4) int width = WIDTH;
	alignas(4) int height = HEIGHT;
	alignas(4) int particleCount = 0;
	alignas(4) int particleDivision = PARTICLE_DIVISION;

};

class Player {
public:
	glm::vec3 position;
	glm::vec3 direction;
	float fovy;
	float aspect;
	float near;
	float far;
	glm::mat4 view;
	glm::mat4 proj;
	const float speed = 30;
	const float rotationSpeed = 0.1f;
	const glm::vec3 up = glm::vec3(0.0f, 0.0f, 1.0f);

	void set(glm::vec3 position, glm::vec3 direction, float fovy, float aspect, float near, float far) {
		this->position = position;
		this->direction = glm::normalize(direction);
		this->fovy = fovy;
		this->aspect = aspect;
		this->near = near;
		this->far = far;

		updateViewMatrix();
		updateProjectionMatrix();
	}

	void updateViewMatrix() {
		view = glm::lookAt(position, position + direction, up);
	}

	void updateProjectionMatrix() {
		proj = glm::perspective(glm::radians(fovy), aspect, near, far);
		proj[1][1] *= -1;
	}

	void keyUpdate(GLFWwindow* window, float deltaTime) {
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			position += direction * speed * deltaTime; updateViewMatrix();
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			position -= direction * speed * deltaTime; updateViewMatrix();
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			position += glm::normalize(glm::cross(up, direction)) * speed * deltaTime; updateViewMatrix();
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			position -= glm::normalize(glm::cross(up, direction)) * speed * deltaTime; updateViewMatrix();
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
			position += glm::normalize(glm::cross(direction, glm::cross(up, direction))) * speed * deltaTime; updateViewMatrix();
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
			position -= glm::normalize(glm::cross(direction, glm::cross(up, direction))) * speed * deltaTime; updateViewMatrix();
		if (glfwGetKey(window, GLFW_KEY_TAB	) == GLFW_PRESS)
			std::cout << "position: ( " << position.x << ", " << position.y << ", " << position.z << " ); direction: ( " 
			<< direction.x << ", " << direction.y << ", " << direction.z << " );" << std::endl;
	}

	void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)) {
			glm::vec2 dMousePos = glm::vec2(xpos, ypos) - oldMousePos;
			rotate(dMousePos.x * rotationSpeed, dMousePos.y * rotationSpeed);
			updateViewMatrix();
		}
		oldMousePos = glm::vec2(xpos, ypos);
	}
private:
	glm::vec2 oldMousePos;
	void rotate(float x, float y) {
		glm::mat4 rotationMat(1);
		rotationMat = glm::rotate(rotationMat, glm::radians(-x), up);
		rotationMat = glm::rotate(rotationMat, glm::radians(y), glm::cross(up, direction));
		direction = glm::vec3(rotationMat * glm::vec4(direction, 1.0));
		direction = glm::normalize(direction);
		if (abs(direction.z) > 0.97f) {
			direction.z = 0.97f * abs(direction.z) / direction.z; 
			float k = direction.x / direction.y;
			direction.y =std::sqrt( (1 - direction.z * direction.z)/(k * k + 1)) * abs(direction.y) / direction.y;
			direction.x = std::sqrt(1 - direction.z * direction.z - direction.y * direction.y) * abs(direction.x) / direction.x;

			//direction = glm::normalize(direction);

		}
	}

};

class App {
public:
	void run() {
		initWindow();
		initVulkan();
		initPlayer();
		mainLoop();
		cleanup();
	}

	void drawing() {

		while (!glfwWindowShouldClose(window)) {
			drawFrame();
		}
	}
private:
	GLFWwindow* window;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;

	VkQueue graphicsQueue;
	VkQueue computeQueue;
	VkQueue presentQueue;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;

	VkRenderPass renderPass;

	VkPipelineLayout graphicsPipelineLayout1;
	VkPipeline graphicsPipeline1;

	VkPipelineLayout computePipelineLayout1;
	VkPipeline computePipeline1;

	VkPipelineLayout computePipelineLayout2;
	VkPipeline computePipeline2;

	VkCommandPool commandPool;

	std::vector<VkImage> depthImagesP1;
	std::vector <VkDeviceMemory> depthImageMemorisP1;
	std::vector<VkImageView> depthImageViewsP1;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	VkDescriptorPool uniformDescriptorPool;
	VkDescriptorSetLayout uniformDescriptorSetLayout;
	std::vector<VkDescriptorSet> uniformDescriptorSets;

	std::vector<VkBuffer> particlesBuffers;
	std::vector<VkDeviceMemory> particlesBuffersMemory;

	std::vector<VkBuffer> particlesDataBuffers;
	std::vector<VkDeviceMemory> particlesDataBuffersMemory;
	std::vector<void*> particlesDataBuffersMapped;

	VkDescriptorPool particlesDescriptorPool;
	VkDescriptorSetLayout particlesDescriptorSetLayout;
	std::vector<VkDescriptorSet> particlesDescriptorSets;

	std::vector<VkCommandBuffer> graphicsCommandBuffers;
	std::vector<VkCommandBuffer> computeCommandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> graphicsInFlightFences;
	std::vector<VkFence> computeInFlightFences;

	//std::thread thread;

	
	int particlesInUse = 0;
	int sortedArray[MAX_PARTICLE_COUNT * 4] = { 0 };

	uint32_t currentFrame = 0;

	int frame = 0;

	static Player player;

	void initWindow() {
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetCursorPosCallback(window, cursorPositionCallback);

	}

	static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos)
	{
		player.cursorPositionCallback(window, xpos, ypos);
	}


	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createUniformDescriptorSetLayout();
		createParticlesDescriptorSetLayout();
		createGraphicsPipeline1();
		createComputePipeline1();
		createComputePipeline2();
		createCommandPool();
		createDepthResources();
		createFramebuffers();
		createUniformBuffers();
		createParticlesBuffers();
		createParticlesDataBuffers();
		createUniformDescriptorPool();
		createParticlesDescriptorPool();
		createUniformDescriptorSets();
		createParticlesDescriptorSets();
		createGraphicsCommandBuffers();
		createComputeCommandBuffers();
		createSyncObjects();
	}


	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_3;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance!");
		}
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;


			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers) {
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}


	void setupDebugMessenger() {
		if (!enableValidationLayers) return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}


	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}


	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);

		bool extensionsSupported = checkDeviceExtensionSupport(device);

		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		VkPhysicalDeviceFeatures supportedFeatures{};
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
				indices.graphicsAndComputeFamily = i;
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

			if (presentSupport) {
				indices.presentFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}


	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.fragmentStoresAndAtomics = VK_TRUE;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.graphicsAndComputeFamily.value(), 0, &computeQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}


	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsAndComputeFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsAndComputeFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}


	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (uint32_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}


	void createRenderPass() {

		VkAttachmentDescription colorAttachmentP1{};
		colorAttachmentP1.format = swapChainImageFormat;
		colorAttachmentP1.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentP1.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentP1.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentP1.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentP1.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentP1.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentP1.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentDescription depthAttachmentP1{};
		depthAttachmentP1.format = findDepthFormat();
		depthAttachmentP1.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachmentP1.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachmentP1.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthAttachmentP1.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachmentP1.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachmentP1.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachmentP1.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;


		VkAttachmentReference colorAttachmentRefP1{};
		colorAttachmentRefP1.attachment = 0;
		colorAttachmentRefP1.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachmentRefP1{};
		depthAttachmentRefP1.attachment = 1;
		depthAttachmentRefP1.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


		VkSubpassDescription subpass1{};
		subpass1.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass1.colorAttachmentCount = 1;
		subpass1.pColorAttachments = &colorAttachmentRefP1;
		subpass1.pDepthStencilAttachment = &depthAttachmentRefP1;

		VkSubpassDependency dependency1{};
		dependency1.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency1.dstSubpass = 0;
		dependency1.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency1.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency1.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		dependency1.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkSubpassDependency dependency2{};
		dependency2.srcSubpass = 0;
		dependency2.dstSubpass = VK_SUBPASS_EXTERNAL;
		dependency2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency2.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependency2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependency2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachmentP1, depthAttachmentP1 };

		std::array<VkSubpassDescription, 1> subpasses = { subpass1 };

		std::array<VkSubpassDependency, 2> subpassDependencies = { dependency1, dependency2 };

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = static_cast<uint32_t>(subpasses.size());
		renderPassInfo.pSubpasses = subpasses.data();
		renderPassInfo.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
		renderPassInfo.pDependencies = subpassDependencies.data();

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}


	}

	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}


	void createUniformDescriptorSetLayout() {
		std::array<VkDescriptorSetLayoutBinding, 1> layoutBindings{};
		layoutBindings[0].binding = 0;
		layoutBindings[0].descriptorCount = 1;
		layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBindings[0].pImmutableSamplers = nullptr;
		layoutBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		layoutInfo.pBindings = layoutBindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &uniformDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute descriptor set layout!");
		}
	}


	void createParticlesDescriptorSetLayout() {
		std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings{};
		layoutBindings[0].binding = 0;
		layoutBindings[0].descriptorCount = 1;
		layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[0].pImmutableSamplers = nullptr;
		layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[1].binding = 1;
		layoutBindings[1].descriptorCount = 1;
		layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[1].pImmutableSamplers = nullptr;
		layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[2].binding = 2;
		layoutBindings[2].descriptorCount = 1;
		layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[2].pImmutableSamplers = nullptr;
		layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		layoutBindings[3].binding = 3;
		layoutBindings[3].descriptorCount = 1;
		layoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		layoutBindings[3].pImmutableSamplers = nullptr;
		layoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
		layoutInfo.pBindings = layoutBindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &particlesDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute descriptor set layout!");
		}
	}


	void createGraphicsPipeline1() {
		auto vertShaderCode = readFile("11/shaders/vertP1.spv");
		auto fragShaderCode = readFile("11/shaders/fragP1.spv");


		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		auto bindingDescription = Particles::getBindingDescription();
		auto attributeDescriptions = Particles::getAttributeDescriptions();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		std::array<VkDescriptorSetLayout, 1> descriptorSetLayouts = { uniformDescriptorSetLayout };

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout1) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = graphicsPipelineLayout1;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline1) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}


	void createComputePipeline1() {
		auto computeShaderCode = readFile("11/shaders/compP1.spv");

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		computeShaderStageInfo.module = computeShaderModule;
		computeShaderStageInfo.pName = "main";

		std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = { uniformDescriptorSetLayout, particlesDescriptorSetLayout };

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout1) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.layout = computePipelineLayout1;
		pipelineInfo.stage = computeShaderStageInfo;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline1) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);
	}


	void createComputePipeline2() {
		auto computeShaderCode = readFile("11/shaders/compP2.spv");

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		computeShaderStageInfo.module = computeShaderModule;
		computeShaderStageInfo.pName = "main";

		std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = { uniformDescriptorSetLayout, particlesDescriptorSetLayout };

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
		pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout2) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		pipelineInfo.layout = computePipelineLayout2;
		pipelineInfo.stage = computeShaderStageInfo;

		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline2) != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);
	}


	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics command pool!");
		}
	}


	void createDepthResources() {
		depthImagesP1.resize(swapChainImageViews.size());
		depthImageMemorisP1.resize(swapChainImageViews.size());
		depthImageViewsP1.resize(swapChainImageViews.size());

		VkFormat depthFormat = findDepthFormat();

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			createImage(swapChainExtent.width, swapChainExtent.height, 1, depthFormat, VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				depthImagesP1[i], depthImageMemorisP1[i]);
			depthImageViewsP1[i] = createImageView(depthImagesP1[i], depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
		}
	}

	void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
		VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = mipLevels;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}


	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageViewsP1[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}


	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				uniformBuffers[i], uniformBuffersMemory[i]);

			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}


	void createParticlesBuffers() {
		std::vector<Particles> particles;
		particles.resize(MAX_PARTICLE_COUNT);

		std::mt19937 gen(0);
		int index = 0;

#if CASE == 0
		for (int i = 0; i < 15; ++i) {
			for (int j = 0; j < 10; ++j) {
				for (int k = 0; k < 15; ++k) {
					float rmin = 2;

					particles[index].position = glm::vec3(i * rmin * 2, j * rmin * std::powf(3, 0.5f), k * rmin * 2 * std::powf(6, 0.5f) / 3);
					particles[index].position.x += rmin * ((j + k) % 2);
					particles[index].position.y += rmin * std::powf(3, 0.5f) / 3 * (k % 2);

					particles[index].lposition = particles[index].position;
					particles[index].type = 0;
					index++;
				}
			}
		}

		for (int i = -5; i < 5; ++i) {
			for (int j = -5; j < 5; ++j) {
				for (int k = -5; k < 5; ++k) {
					float rmin = 1;

					particles[index].position = glm::vec3(i * rmin * 2, j * rmin * std::powf(3, 0.5f), k * rmin * 2 * std::powf(6, 0.5f) / 3);
					particles[index].position.x += rmin * ((j + k) % 2);
					particles[index].position.y += rmin * std::powf(3, 0.5f) / 3 * (k % 2);
					if (glm::length(particles[index].position) / rmin > 6)continue;
					particles[index].position += glm::vec3(30, -40, 27);


					particles[index].lposition = particles[index].position;
					particles[index].lposition += glm::vec3(0, -0.1f, 0);
					particles[index].type = 1;
					index++;
				}
			}
		}
#elif CASE == 1

		for (int i = 0; i < 50; ++i) {
			for (int j = 0; j < 50; ++j) {
				for (int k = 0; k < 1; ++k) {
					float rmin = 2;

					particles[index].position = glm::vec3(i * rmin * 2, j * rmin * std::powf(3, 0.5f), k * rmin * 2 * std::powf(6, 0.5f) / 3);
					particles[index].position.x += rmin * ((j + k) % 2);
					particles[index].position.y += rmin * std::powf(3, 0.5f) / 3 * (k % 2);

					particles[index].position += glm::vec3(-50, -50, 0);


					particles[index].lposition = particles[index].position;
					particles[index].type = 0;
					index++;
				}
			}
		}

		for (int i = 0; i < 10; ++i) {
			for (int j = 0; j < 10; ++j) {
				for (int k = 0; k < 10; ++k) {
					float rmin = 2;

					particles[index].position = glm::vec3(i * rmin * 2, j * rmin * std::powf(3, 0.5f), k * rmin * 2 * std::powf(6, 0.5f) / 3);
					particles[index].position.x += rmin * ((j + k) % 2);
					particles[index].position.y += rmin * std::powf(3, 0.5f) / 3 * (k % 2);

					particles[index].position += glm::vec3(0, 0, 100);


					particles[index].lposition = particles[index].position;
					particles[index].type = 1;
					index++;
				}
			}
		}
#elif CASE == 2
		std::uniform_real_distribution<> dist(-1, 1);
		// 20 20 20
		for (int i = 0; i < 20; ++i) {
			for (int j = 0; j < 20; ++j) {
				for (int k = 0; k < 25; ++k) {
					float rmin = 2;

					particles[index].position = glm::vec3(i * rmin * 2, j * rmin * std::powf(3, 0.5f), k * rmin * 2 * std::powf(6, 0.5f) / 3);
					particles[index].position.x += rmin * ((j + k) % 2);
					particles[index].position.y += rmin * std::powf(3, 0.5f) / 3 * (k % 2);


					particles[index].lposition = particles[index].position;
					//particles[index].lposition += +glm::vec3(dist(gen) / 30, dist(gen) / 30, dist(gen) / 30);
					particles[index].type = 0;
					index++;
				}
			}
		}
#endif
		particlesInUse = index;

		for (int i = 0; i < particlesInUse; i++)
		{
			sortedArray[i * 4] = int(std::ceil(particles[i].position.x / PARTICLE_DIVISION));
			sortedArray[i * 4 + 1] = int(std::ceil(particles[i].position.y / PARTICLE_DIVISION));
			sortedArray[i * 4 + 2] = int(std::ceil(particles[i].position.z / PARTICLE_DIVISION));
			sortedArray[i * 4 + 3] = i;
		}

		std::qsort(sortedArray, particlesInUse, sizeof(unsigned int) * 4, Compar);

		

		VkDeviceSize bufferSize = sizeof(Particles) * MAX_PARTICLE_COUNT;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, particles.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		particlesBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		particlesBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				particlesBuffers[i], particlesBuffersMemory[i]);
			copyBuffer(stagingBuffer, particlesBuffers[i], bufferSize);
		}

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}


	void createParticlesDataBuffers() {
		VkDeviceSize bufferSize = sizeof(unsigned int) * MAX_PARTICLE_COUNT * 4;

		particlesDataBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		particlesDataBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		particlesDataBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				particlesDataBuffers[i], particlesDataBuffersMemory[i]);
			vkMapMemory(device, particlesDataBuffersMemory[i], 0, bufferSize, 0, &particlesDataBuffersMapped[i]);
			memcpy(particlesDataBuffersMapped[i], sortedArray, bufferSize);
		}
	}


	void createUniformDescriptorPool() {
		std::array<VkDescriptorPoolSize, 1> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &uniformDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}


	void createParticlesDescriptorPool() {
		std::array<VkDescriptorPoolSize, 4> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &particlesDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}


	void createUniformDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, uniformDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = uniformDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		uniformDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, uniformDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			std::array<VkWriteDescriptorSet, 1> descriptorWrites{};

			VkDescriptorBufferInfo uniformBufferInfo{};
			uniformBufferInfo.buffer = uniformBuffers[i];
			uniformBufferInfo.offset = 0;
			uniformBufferInfo.range = sizeof(UniformBufferObject);

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = uniformDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &uniformBufferInfo;


			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}


	void createParticlesDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, particlesDescriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = particlesDescriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		allocInfo.pSetLayouts = layouts.data();

		particlesDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &allocInfo, particlesDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			std::array<VkWriteDescriptorSet, 4> descriptorWrites{};

			VkDescriptorBufferInfo  particlesBufferInfoLastFrame{};
			particlesBufferInfoLastFrame.buffer = particlesBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT];
			particlesBufferInfoLastFrame.offset = 0;
			particlesBufferInfoLastFrame.range = sizeof(Particles) * MAX_PARTICLE_COUNT;

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = particlesDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &particlesBufferInfoLastFrame;

			VkDescriptorBufferInfo particlesBufferInfoCurrentFrame{};
			particlesBufferInfoCurrentFrame.buffer = particlesBuffers[i];
			particlesBufferInfoCurrentFrame.offset = 0;
			particlesBufferInfoCurrentFrame.range = sizeof(Particles) * MAX_PARTICLE_COUNT;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = particlesDescriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &particlesBufferInfoCurrentFrame;

			VkDescriptorBufferInfo particlesDataBufferInfoCurrentFrame{};
			particlesDataBufferInfoCurrentFrame.buffer = particlesDataBuffers[i];
			particlesDataBufferInfoCurrentFrame.offset = 0;
			particlesDataBufferInfoCurrentFrame.range = sizeof(unsigned int) * MAX_PARTICLE_COUNT * 4;

			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = particlesDescriptorSets[i];
			descriptorWrites[2].dstBinding = 2;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pBufferInfo = &particlesDataBufferInfoCurrentFrame;

			VkDescriptorBufferInfo particlesDataBufferInfoNextFrame{};
			particlesDataBufferInfoNextFrame.buffer = particlesDataBuffers[(i + 1) % MAX_FRAMES_IN_FLIGHT];
			particlesDataBufferInfoNextFrame.offset = 0;
			particlesDataBufferInfoNextFrame.range = sizeof(unsigned int) * MAX_PARTICLE_COUNT * 4;

			descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[3].dstSet = particlesDescriptorSets[i];
			descriptorWrites[3].dstBinding = 3;
			descriptorWrites[3].dstArrayElement = 0;
			descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[3].descriptorCount = 1;
			descriptorWrites[3].pBufferInfo = &particlesDataBufferInfoNextFrame;


			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}


	void createGraphicsCommandBuffers() {
		graphicsCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)graphicsCommandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, graphicsCommandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}


	void createComputeCommandBuffers() {
		computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)computeCommandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate compute command buffers!");
		}
	}


	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		graphicsInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &graphicsInFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create graphics synchronization objects for a frame!");
			}
			if (vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create compute synchronization objects for a frame!");
			}
		}
	}


	void initPlayer() {
#if CASE == 0
		player.set(glm::vec3(128.677, -215.75, 64.9492), glm::vec3(-0.394279, 0.893956, -0.213041), 45, swapChainExtent.width / (float)swapChainExtent.height, 10, 1000);
#elif CASE == 1
		player.set(glm::vec3(282.141f, -138.19f, 162.473f), glm::vec3(-0.674087f, 0.519118f, -0.525474f), 45, swapChainExtent.width / (float)swapChainExtent.height, 10, 1000);
#elif CASE == 2
		player.set(glm::vec3(282.141f, -138.19f, 162.473f), glm::vec3(-0.674087f, 0.519118f, -0.525474f), 45, swapChainExtent.width / (float)swapChainExtent.height, 10, 1000);
#else 
		player.set(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1), 45, swapChainExtent.width / (float)swapChainExtent.height, 10, 1000);
#endif
	}


	void mainLoop() {
		//thread = std::thread([this] {this->drawing(); });
		auto start = std::chrono::steady_clock::now();
		std::chrono::steady_clock::duration deltaTime;

		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			player.keyUpdate(window, (std::chrono::duration_cast<std::chrono::microseconds>(deltaTime) / 1000000.0f).count());
			drawFrame();

#if STOP_EVERY_FRAME == 1
			char a;
			std::cin >> a;
#endif

			auto end = std::chrono::steady_clock::now();
			deltaTime = end - start;

#if SHOW_FPS == 1
			if (frame % 10 == 1)
				std::cout << 1.0f / (std::chrono::duration_cast<std::chrono::microseconds>(deltaTime) / 1000000.0f).count() << std::endl;
#endif
			start = end;
			frame += 1;
		}
		//thread.join();
		vkDeviceWaitIdle(device);
	}

	void drawFrame() {
		vkWaitForFences(device, 1, &computeInFlightFences[(currentFrame - 1) % MAX_FRAMES_IN_FLIGHT], VK_TRUE, UINT64_MAX);

		vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &computeInFlightFences[currentFrame]);

		updateBuffers();

		vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
		recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);

		VkSubmitInfo submitInfoComp{};
		submitInfoComp.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfoComp.commandBufferCount = 1;
		submitInfoComp.pCommandBuffers = &computeCommandBuffers[currentFrame];

		if (vkQueueSubmit(computeQueue, 1, &submitInfoComp, computeInFlightFences[currentFrame]) != VK_SUCCESS)
			throw std::runtime_error("failed to submit compute command buffer!");



		vkWaitForFences(device, 1, &graphicsInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &graphicsInFlightFences[currentFrame]);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);


		if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
			throw std::runtime_error("failed to acquire swap chain image!");



		vkResetCommandBuffer(graphicsCommandBuffers[currentFrame], 0);
		recordGraphicsCommandBuffer(graphicsCommandBuffers[currentFrame], imageIndex);

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

		VkSubmitInfo submitInfoGraph{};
		submitInfoGraph.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfoGraph.waitSemaphoreCount = 1;
		submitInfoGraph.pWaitSemaphores = waitSemaphores;
		submitInfoGraph.pWaitDstStageMask = waitStages;
		submitInfoGraph.commandBufferCount = 1;
		submitInfoGraph.pCommandBuffers = &graphicsCommandBuffers[currentFrame];
		submitInfoGraph.signalSemaphoreCount = 1;
		submitInfoGraph.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfoGraph, graphicsInFlightFences[currentFrame]) != VK_SUCCESS)
			throw std::runtime_error("failed to submit draw command buffer!");



		VkSwapchainKHR swapChains[] = { swapChain };

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrame];
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);


		if (result != VK_SUCCESS)
			throw std::runtime_error("failed to present swap chain image!");



		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateBuffers() {
		UniformBufferObject ubo{};
		ubo.particleCount = particlesInUse;
		ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(0.03f) * 0, glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = player.view;
		ubo.proj = player.proj;

		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));


		VkDeviceSize bufferSize1 = sizeof(unsigned int) * particlesInUse * 4;

		memcpy( sortedArray, particlesDataBuffersMapped[currentFrame], bufferSize1);


		std::qsort(sortedArray, particlesInUse, sizeof(unsigned int) * 4, Compar);


		memcpy(particlesDataBuffersMapped[currentFrame], sortedArray, bufferSize1);
	}

	void recordComputeCommandBuffer(VkCommandBuffer commandBuffer) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording compute command buffer!");
		}

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline1);
		std::array<VkDescriptorSet, 2> dSetsP1 = { uniformDescriptorSets[currentFrame], particlesDescriptorSets[currentFrame] };
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout1, 0, 2, dSetsP1.data(), 0, nullptr);
		vkCmdDispatch(commandBuffer, particlesInUse, 1, 1);
		
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline2);
		std::array<VkDescriptorSet, 2> dSetsP2 = { uniformDescriptorSets[currentFrame], particlesDescriptorSets[currentFrame] };
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout2, 0, 2, dSetsP2.data(), 0, nullptr);
		vkCmdDispatch(commandBuffer, particlesInUse, 1, 1);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record compute command buffer!");
		}
	}

	void recordGraphicsCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 0.0f} };
		clearValues[1].depthStencil = { 1.0f, 0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		{
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float)swapChainExtent.width;
			viewport.height = (float)swapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

			VkRect2D scissor{};
			scissor.offset = { 0, 0 };
			scissor.extent = swapChainExtent;
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);


			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline1);

			std::array<VkDescriptorSet, 1> dSetsP1 = { uniformDescriptorSets[currentFrame] };
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout1, 0, 1, dSetsP1.data(), 0, nullptr);

			VkDeviceSize offsets[] = { 0 };
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, &particlesBuffers[currentFrame], offsets);

			vkCmdDraw(commandBuffer, particlesInUse, 1, 0, 0);
		}
		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}


	void cleanup() {
		for (int i = 0; i < swapChainImageViews.size(); i++)
		{
			vkDestroyImageView(device, depthImageViewsP1[i], nullptr);
			vkDestroyImage(device, depthImagesP1[i], nullptr);
			vkFreeMemory(device, depthImageMemorisP1[i], nullptr);
		}

		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);


		vkDestroyPipeline(device, graphicsPipeline1, nullptr);
		vkDestroyPipelineLayout(device, graphicsPipelineLayout1, nullptr);

		vkDestroyPipeline(device, computePipeline1, nullptr);
		vkDestroyPipelineLayout(device, computePipelineLayout1, nullptr);

		vkDestroyPipeline(device, computePipeline2, nullptr);
		vkDestroyPipelineLayout(device, computePipelineLayout2, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, particlesBuffers[i], nullptr);
			vkFreeMemory(device, particlesBuffersMemory[i], nullptr);

			vkDestroyBuffer(device, particlesDataBuffers[i], nullptr);
			vkFreeMemory(device, particlesDataBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, particlesDescriptorPool, nullptr);
		vkDestroyDescriptorPool(device, uniformDescriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(device, particlesDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, uniformDescriptorSetLayout, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);

			vkDestroyFence(device, graphicsInFlightFences[i], nullptr);
			vkDestroyFence(device, computeInFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
};
Player App::player = Player();


int main() {

#ifdef _DEBUG
	system("11\\shaders\\compile.bat");
#endif
	App app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}